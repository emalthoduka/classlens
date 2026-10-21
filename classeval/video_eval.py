"""Video quality from container metadata + a small sample of frames.

Only ~16 general frames and ~20 'teacher is speaking' frames are decoded, each
via fast seeking, so a 1-hour recording costs a few seconds, not a full decode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import background, config, media
from .transcript import Transcript, fmt_ts

Progress = Callable[[float, str], None]


def _interp(x, xs, ys) -> float:
    return float(np.interp(x, xs, ys))


def _metric(name: str, score: float, value: str, explanation: str, **extra) -> dict:
    return {"name": name, "score": round(float(np.clip(score, 0, 100)), 1), "value": value, "explanation": explanation, **extra}


def teacher_times(tr: Transcript, n: int) -> list[float]:
    """Mid-points of teacher turns (>= 2 s), spread evenly across the class."""
    segs = [s for s in tr.teacher_segments() if s.duration >= 2.0] or tr.teacher_segments()
    if not segs:
        return []
    idx = np.unique(np.linspace(0, len(segs) - 1, min(n, len(segs))).round().astype(int))
    return [round(segs[i].start + segs[i].duration / 2, 2) for i in idx]


def _blockiness(g: np.ndarray) -> float:
    def count(arr: np.ndarray, phase: int) -> int:
        d = np.abs(np.diff(arr, axis=1))
        if d.shape[1] < 16:
            return 0
        cols = np.arange(phase, d.shape[1] - 1, 8)
        cols = cols[cols >= 1]
        step = d[:, cols]
        neigh = (d[:, cols - 1] + d[:, cols + 1]) / 2
        return int(((step >= 1.5) & (step <= 12) & (neigh < 0.75)).sum())

    on = count(g, 7) + count(g.T, 7)
    off = count(g, 3) + count(g.T, 3)
    return float((on + 20) / (off + 20))


def frame_quality(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] != 1280:  # normalise so thresholds are resolution-independent
        gray = cv2.resize(gray, (1280, int(gray.shape[0] * 1280 / gray.shape[1])), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    # Immerkaer noise estimate restricted to flat areas (edges would inflate it).
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(cv2.filter2D(g, -1, kernel))
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    flat = np.hypot(gx, gy) < 20
    noise = float(np.sqrt(np.pi / 2) * resp[flat].mean() / 6) if flat.sum() > 1000 else float("nan")
    # Blockiness: small luminance steps on the 8-px coding grid inside otherwise smooth
    # areas, relative to the same pattern off-grid. Strong real edges (e.g. chessboard
    # squares that happen to align with the grid) are excluded by the step-size limit.
    raw = img if img.shape[1] <= 1920 else cv2.resize(img, (1920, int(img.shape[0] * 1920 / img.shape[1])))
    rg = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    block = _blockiness(rg)
    mean = float(gray.mean())
    clipped = float(((gray < 4) | (gray > 251)).mean())
    black = mean < 10 and gray.std() < 6
    return {"sharpness": lap, "noise": noise, "blockiness": block, "brightness": mean, "clipped": clipped, "black": bool(black)}


def analyze(video: Path, info: dict, tr: Transcript, work: Path, thumbs_dir: Path, progress: Progress) -> dict:
    dur = info.get("duration") or tr.duration
    general = [round(float(t), 2) for t in np.linspace(dur * 0.03, dur * 0.97, config.VIDEO_SAMPLE_FRAMES)] if dur else []
    teach = teacher_times(tr, config.TEACHER_FRAMES)
    frames = media.grab_frames(video, sorted(set(general + teach)), work / "frames", lambda p, d: progress(p * 0.6, d))

    metrics: dict[str, dict] = {}
    w, h = info.get("width", 0), info.get("height", 0)
    short = min(w, h)
    ar = w / h if h else 0
    ar_note = "16:9" if abs(ar - 16 / 9) < 0.03 else "4:3" if abs(ar - 4 / 3) < 0.03 else f"{ar:.2f}:1"
    metrics["resolution"] = _metric(
        "Resolution", _interp(short, [240, 360, 480, 720, 1080], [10, 35, 55, 85, 100]), f"{w}×{h} ({ar_note})",
        f"Recorded at {w}×{h}, aspect {ar_note}. "
        + ("Full HD - crisp board and text." if short >= 1080 else "HD - adequate for board sharing." if short >= 720
           else "Below HD - small text on shared boards may be hard to read."),
    )
    fps = info.get("avg_fps") or info.get("r_fps") or 0
    metrics["framerate"] = _metric(
        "Frame rate", _interp(fps, [5, 10, 15, 24], [10, 40, 70, 100]), f"{fps:.1f} fps",
        f"Average {fps:.1f} frames/s - " + ("smooth motion." if fps >= 24 else "acceptable for lessons." if fps >= 15
                                               else "choppy; moves and gestures may stutter."),
    )
    vbr = info.get("video_bitrate") or max(info.get("container_bitrate", 0) - info.get("audio_bitrate", 0), 0)
    bpp = vbr / (w * h * fps) if w and h and fps else 0.0
    q = []
    for t in general:
        p = frames.get(t)
        if p is not None:
            img = cv2.imread(str(p))
            if img is not None:
                q.append(frame_quality(img))
    progress(0.7, "Analysing frame quality…")
    if q:
        lap = float(np.median([x["sharpness"] for x in q]))
        noise_vals = [x["noise"] for x in q if np.isfinite(x["noise"])]
        nz = float(np.median(noise_vals)) if noise_vals else float("nan")
        blk = float(np.median([x["blockiness"] for x in q]))
        bright = float(np.median([x["brightness"] for x in q]))
        clip = float(np.median([x["clipped"] for x in q]))
        black = sum(x["black"] for x in q)
        metrics["sharpness"] = _metric(
            "Sharpness", _interp(lap, [10, 30, 80, 200], [5, 35, 70, 100]), f"Laplacian var {lap:.0f}",
            "Image detail is " + ("crisp." if lap >= 150 else "acceptable." if lap >= 60 else "soft / blurry (focus or heavy compression)."),
        )
        metrics["noise"] = _metric(
            "Noise / grain", 80 if not np.isfinite(nz) else _interp(nz, [2, 4, 7, 12], [100, 75, 45, 10]),
            "n/a" if not np.isfinite(nz) else f"σ ≈ {nz:.1f}",
            "Not measurable (few flat regions)." if not np.isfinite(nz) else
            ("Clean image." if nz < 3 else "Some grain (low light / sensor noise)." if nz < 6 else "Grainy image - usually poor lighting."),
        )
        metrics["compression"] = _metric(
            "Compression artefacts", _interp(np.log10(max(blk, 1.0)), [0.5, 1.0, 1.6, 2.3], [100, 70, 40, 10]),
            f"block ratio {blk:.1f}",
            "No visible blocking." if blk < 3 else "Mild 8×8 blocking in flat areas." if blk < 15 else "Visible blocking artefacts.",
        )
        exp_score = 100 - _interp(abs(bright - 115), [40, 70, 100], [0, 40, 80]) - _interp(clip, [0.05, 0.2, 0.5], [0, 30, 60])
        metrics["exposure"] = _metric(
            "Exposure / brightness", exp_score, f"mean {bright:.0f}/255, {clip:.0%} clipped",
            f"Median brightness {bright:.0f}/255 with {clip:.0%} of pixels crushed or blown out."
            + (f" {black} of {len(q)} sampled frames were black (no video)." if black else ""),
            black_frames=int(black),
        )
    else:
        for k, name in (("sharpness", "Sharpness"), ("noise", "Noise / grain"), ("compression", "Compression artefacts"),
                        ("exposure", "Exposure / brightness")):
            metrics[k] = _metric(name, 50, "n/a", "Frames could not be decoded.")

    # Low bitrate only matters if it shows: static screen-shares legitimately compress
    # to very little, so the visible-artefact score acts as a floor.
    bit_curve = _interp(bpp, [0.005, 0.01, 0.02, 0.04, 0.08], [10, 30, 55, 80, 100])
    shown = metrics["compression"]["score"] if q else 50
    metrics["bitrate"] = _metric(
        "Bitrate / compression budget", max(bit_curve, 0.85 * shown),
        f"{vbr / 1000:,.0f} kbps ({bpp:.3f} bits/pixel) {info.get('video_codec', '')}",
        f"{vbr / 1000:,.0f} kbps for {w}×{h}@{fps:.0f} fps = {bpp:.3f} bits per pixel. "
        + ("Generous - few artefacts expected." if bpp >= 0.06 else "Typical video-call budget." if bpp >= 0.02
           else "Low budget" + (", but sampled frames show no visible artefacts (mostly static content)." if shown >= 80
                                else " and visible artefacts in sampled frames.")),
    )

    # ---- Brand background while the teacher speaks ------------------------------
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    for i, t in enumerate(teach):
        p = frames.get(t)
        img = cv2.imread(str(p)) if p is not None else None
        if img is None:
            continue
        m = background.detect(img)
        thumb = thumbs_dir / f"teacher_{i:02d}_{'yes' if m.found else 'no'}.jpg"
        cv2.imwrite(str(thumb), background.annotate(img, m), [cv2.IMWRITE_JPEG_QUALITY, 80])
        checks.append({"t": t, "timestamp": fmt_ts(t), "found": m.found, "inliers": m.inliers, "ncc": m.ncc, "thumb": str(thumb)})
        progress(0.7 + 0.3 * (i + 1) / max(len(teach), 1), f"Checking brand background {i + 1}/{len(teach)}")
    if checks:
        hit = sum(c["found"] for c in checks)
        yes = [c["timestamp"] for c in checks if c["found"]]
        no = [c["timestamp"] for c in checks if not c["found"]]
        # Seeing the background a few times is enough proof it was used (the teacher tile is
        # often hidden or tiny during screen share), so full marks from BRAND_FULL_MARKS_HITS.
        full = config.BRAND_FULL_MARKS_HITS
        verdict = (" Seen more than a couple of times - full marks." if hit >= full
                   else f" Full marks need {full}+ sightings." if hit else "")
        metrics["brand_background"] = _metric(
            "Branded background (teacher cam)", 100.0 * min(hit, full) / full,
            f"{hit}/{len(checks)} teacher frames",
            f"The academy background was visible in {hit} of {len(checks)} frames sampled while the teacher was speaking."
            + verdict
            + (f" Seen at {', '.join(yes[:6])}." if yes else "")
            + (f" Not found at {', '.join(no[:6])} (camera off, different background, or tile too small)." if no else ""),
            checks=checks,
        )
    else:
        metrics["brand_background"] = _metric(
            "Branded background (teacher cam)", 0, "no frames",
            "No frames could be sampled during teacher speech, so the background could not be verified.", checks=[],
        )

    wts = config.VIDEO_WEIGHTS
    score = sum(metrics[k]["score"] * wts[k] for k in wts) / sum(wts.values())
    for k in metrics:
        metrics[k]["weight"] = wts[k]
    readable = {
        "resolution": f"{w}×{h}",
        "frame_rate": f"{fps:.1f} fps",
        "codec": info.get("video_codec", ""),
        "video_bitrate": f"{vbr / 1000:,.0f} kbps",
        "duration": f"{(info.get('duration') or 0) / 60:.1f} min",
        "file_size": f"{(info.get('size_bytes') or 0) / 1024**2:,.0f} MB",
        "frames_sampled": f"{len(general)} general + {len(teach)} teacher-speaking",
    }
    return {"score": round(score, 1), "metrics": metrics, "info": readable}
