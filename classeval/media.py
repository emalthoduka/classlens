"""ffmpeg / ffprobe wrappers. All work streams from disk; nothing loads a whole video."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .errors import UserFacingError

Progress = Callable[[float, str], None]


def _bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    ff_dir = os.getenv("FFMPEG_DIR")
    if ff_dir:
        for cand in (Path(ff_dir) / name, Path(ff_dir) / f"{name}.exe"):
            if cand.exists():
                return str(cand)
    raise UserFacingError(f"'{name}' is not installed on the server (add ffmpeg to packages.txt).")


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout, check=False)


def probe(path: str | Path) -> dict:
    """Container/stream metadata via ffprobe (no decoding)."""
    res = _run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=120,
    )
    if res.returncode != 0:
        raise UserFacingError("Could not read this file as a video. Is it a valid recording (mp4/mkv/webm/mov)?")
    data = json.loads(res.stdout or b"{}")
    fmt = data.get("format", {})
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    def _rate(r: str | None) -> float:
        try:
            n, d = (r or "0/1").split("/")
            return float(n) / float(d) if float(d) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    info = {
        "duration": float(fmt.get("duration") or 0.0),
        "size_bytes": int(fmt.get("size") or Path(path).stat().st_size),
        "format_name": fmt.get("format_name", ""),
        "container_bitrate": int(fmt.get("bit_rate") or 0),
        "has_video": v is not None,
        "has_audio": a is not None,
    }
    if v:
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        # Respect rotation metadata (phone recordings).
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = abs(int(sd["rotation"]))
        if rot in (90, 270):
            w, h = h, w
        info.update(
            width=w,
            height=h,
            video_codec=v.get("codec_name", ""),
            avg_fps=round(_rate(v.get("avg_frame_rate")), 2),
            r_fps=round(_rate(v.get("r_frame_rate")), 2),
            video_bitrate=int(v.get("bit_rate") or 0),
            pix_fmt=v.get("pix_fmt", ""),
            nb_frames=int(v.get("nb_frames") or 0),
        )
    if a:
        info.update(
            audio_codec=a.get("codec_name", ""),
            audio_sample_rate=int(a.get("sample_rate") or 0),
            audio_channels=int(a.get("channels") or 0),
            audio_bitrate=int(a.get("bit_rate") or 0),
        )
    return info


def _ffmpeg_with_progress(args: list[str], duration: float, progress: Progress, label: str) -> None:
    """Run ffmpeg and report progress parsed from `-progress pipe:1`."""
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("out_time_us=") and duration > 0:
            try:
                t = int(line.split("=", 1)[1]) / 1e6
            except ValueError:
                continue
            progress(min(t / duration, 1.0), f"{label} {min(t / duration, 1.0):.0%}")
    _, err = proc.communicate()
    if proc.returncode != 0:
        tail = (err or "").strip().splitlines()[-3:]
        raise UserFacingError(f"{label} failed (ffmpeg): {' | '.join(tail)}")


def extract_audio(video: Path, out_dir: Path, duration: float, progress: Progress) -> tuple[Path, Path]:
    """One decode pass, two outputs:

    * analysis.wav - 16 kHz mono PCM for signal analysis (~115 MB/hour)
    * api.mp3      - 16 kHz mono 48 kbps for the transcription API (~21 MB/hour)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav, mp3 = out_dir / "analysis.wav", out_dir / "api.mp3"
    args = [
        "-i", str(video), "-vn", "-sn", "-dn",
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k", str(mp3),
    ]  # fmt: skip
    _ffmpeg_with_progress(args, duration, progress, "Extracting audio")
    return wav, mp3


def cut_audio(src: Path, start: float, end: float, dest: Path, codec_args: list[str] | None = None) -> Path:
    codec_args = codec_args or ["-c", "copy"]
    res = _run(
        [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(src), *codec_args, str(dest)],
        timeout=300,
    )  # fmt: skip
    if res.returncode != 0 or not dest.exists():
        raise UserFacingError("Failed to split audio for transcription.")
    return dest


def plan_chunks(duration: float, silences: list[tuple[float, float]], target: float, search: float | None = None) -> list[tuple[float, float]]:
    """Split [0, duration] into ~target-second chunks, cutting inside silences when possible."""
    search = search if search is not None else min(45.0, target / 4)
    bounds = [0.0]
    while duration - bounds[-1] > target * 1.25:
        ideal = bounds[-1] + target
        best = None
        for s, e in silences:
            mid = (s + e) / 2
            if abs(mid - ideal) <= search and (best is None or abs(mid - ideal) < abs(best - ideal)):
                best = mid
        bounds.append(best if best is not None else ideal)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def frame_at(video: Path, t: float, dest: Path, max_width: int = 1280) -> Path | None:
    """Grab one frame using fast input seeking (decodes only from the nearest keyframe)."""
    res = _run(
        [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-ss", f"{max(t, 0):.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale='min({max_width},iw)':-2", "-q:v", "2", str(dest)],
        timeout=120,
    )  # fmt: skip
    return dest if res.returncode == 0 and dest.exists() and dest.stat().st_size > 0 else None


def grab_frames(
    video: Path, times: list[float], out_dir: Path, progress: Progress, prefix: str = "f", max_width: int = 1920
) -> dict[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[float, Path] = {}
    done = 0

    def _one(t: float):
        return t, frame_at(video, t, out_dir / f"{prefix}_{int(t * 1000):010d}.jpg", max_width)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for t, p in ex.map(_one, times):
            done += 1
            if p:
                results[t] = p
            progress(done / max(len(times), 1), f"Sampling frames {done}/{len(times)}")
    return results
