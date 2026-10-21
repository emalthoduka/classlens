"""Build a synthetic 'class recording' (mp4 + matching VTT) for offline pipeline tests.

Usage: python -m tests.make_sample OUT_DIR
Speech is produced by Windows SAPI text-to-speech if available, otherwise by
modulated tones (still exercises every pipeline stage).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from classeval import media
from tests import synth

SR = 16000
LINES = [
    ("Teacher", "Hello everyone! How was your day? Today we will learn the back rank mate."),
    ("Gautham", "Hi ma'am, it was good."),
    ("Teacher", "Very good. Do you remember what we learned last class about checks?"),
    ("Nisha", "Yes ma'am, a check attacks the king."),
    ("Teacher", "Excellent, Nisha! Now look at this board. White to play. What is the best move?"),
    ("Gautham", "Rook to e8 is checkmate."),
    ("Teacher", "Perfect, Gautham. The king is trapped by its own pawns on the second rank."),
    ("Teacher", "For homework, please solve five back rank puzzles on lichess. Bye everyone, see you on Thursday!"),
]


def tts(text: str, voice_hint: str, out: Path) -> bool:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$v=$s.GetInstalledVoices()|%{{$_.VoiceInfo.Name}}|?{{$_ -like '*{voice_hint}*'}}|select -first 1;"
        "if($v){$s.SelectVoice($v)};"
        f"$s.SetOutputToWaveFile('{out}');$s.Speak(\"{text}\");$s.Dispose()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True, timeout=60)
        return out.exists()
    except (OSError, subprocess.SubprocessError):
        return False


def tone_speech(seconds: float, f0: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    syll = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) ** 2
    return 0.3 * syll * np.sin(2 * np.pi * f0 * t + 3 * np.sin(2 * np.pi * 5 * t))


def load_mono16k(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        idx = np.arange(0, len(x), sr / SR)
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def fmt(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main(out_dir: str) -> None:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pieces, cues, t = [], [], 2.0
    pieces.append(np.zeros(int(t * SR), np.float32))
    for i, (who, text) in enumerate(LINES):
        wav = out / f"line_{i}.wav"
        voice = "Zira" if who == "Teacher" else "David"
        x = load_mono16k(wav) if tts(text, voice, wav) else tone_speech(1.5 + len(text) / 25, 220 if who == "Teacher" else 140)
        x = x / (np.abs(x).max() + 1e-9) * (0.5 if who == "Teacher" else 0.35)
        cues.append((t, t + len(x) / SR, "Ayswarya Joseph" if who == "Teacher" else who, text))
        pieces += [x, np.zeros(int(1.2 * SR), np.float32)]
        t += len(x) / SR + 1.2
    audio = np.concatenate(pieces)
    n = np.arange(len(audio)) / SR
    audio = audio + 0.003 * rng.standard_normal(len(audio)) + 0.004 * np.sin(2 * np.pi * 50 * n)
    sf.write(out / "audio.wav", audio.astype(np.float32), SR)
    dur = len(audio) / SR

    vtt = ["WEBVTT", ""]
    for i, (s, e, who, text) in enumerate(cues, 1):
        vtt += [str(i), f"{fmt(s)} --> {fmt(e)}", f"{who}: {text}", ""]
    (out / "class.vtt").write_text("\n".join(vtt), encoding="utf-8")

    # Video: gallery with branded teacher for first half, screen share for second half,
    # final 15 % shows the teacher without the brand (camera background changed).
    W, H = 1280, 720
    scenes = [(0.0, synth.degrade(synth.gallery(W, H, True))),
              (0.5, synth.degrade(synth.screenshare(W, H, True, 320))),
              (0.85, synth.degrade(synth.gallery(W, H, False)))]
    lst = []
    for i, (start, img) in enumerate(scenes):
        p = out / f"scene_{i}.png"
        cv2.imwrite(str(p), img)
        end = scenes[i + 1][0] if i + 1 < len(scenes) else 1.0
        lst += [f"file '{p.as_posix()}'", f"duration {(end - start) * dur:.3f}"]
    lst.append(f"file '{(out / f'scene_{len(scenes) - 1}.png').as_posix()}'")
    (out / "scenes.txt").write_text("\n".join(lst), encoding="utf-8")
    subprocess.run(
        [media._bin("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(out / "scenes.txt"), "-i", str(out / "audio.wav"), "-vf", "fps=25,format=yuv420p",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "96k", "-shortest",
         str(out / "class.mp4")],
        check=True,
    )  # fmt: skip
    print(f"wrote {out / 'class.mp4'} ({dur:.1f}s) and {out / 'class.vtt'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/synthetic")
