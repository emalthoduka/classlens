"""Signal-level audio quality analysis (no ML, no API calls).

The WAV is read in 10-second blocks, never loaded whole. Pass 1 (envelope) runs
before transcription and also yields silences used to choose chunk boundaries.
Pass 2 (spectra) runs after transcription so it can use the transcript to tell
speech apart from background - which makes SNR / noise estimates far more
reliable than blind voice-activity detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from . import config
from .transcript import Transcript, fmt_ts

Progress = Callable[[float, str], None]

SR = 16000
F10 = 160  # 10 ms
F100 = 1600  # 100 ms
BLOCK = SR * 10
EPS = 1e-12
DIGITAL_SILENCE_DB = -90.0


def _db(x):
    return 10 * np.log10(np.maximum(x, EPS))


def k_weighting(fs: int = SR):
    """ITU-R BS.1770 K-weighting as two biquads (generic design valid at any fs)."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / fs)
    Vh = 10 ** (G / 20)
    Vb = Vh**0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = np.array([(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0])
    a1 = np.array([1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / fs)
    a0 = 1 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    return (b1, a1), (b2, a2)


@dataclass
class Envelope:
    ms10: np.ndarray  # mean square per 10 ms frame
    kw100: np.ndarray  # K-weighted mean square per 100 ms frame
    peak: float
    clip_samples: int
    clip_events: int
    n_samples: int

    @property
    def db10(self) -> np.ndarray:
        return _db(self.ms10)


def envelope(wav: Path, progress: Progress) -> Envelope:
    info = sf.info(str(wav))
    if info.samplerate != SR:
        raise ValueError("analysis wav must be 16 kHz")
    (b1, a1), (b2, a2) = k_weighting()
    z1 = np.zeros(2)
    z2 = np.zeros(2)
    ms10, kw100 = [], []
    peak, clips, clip_events, n = 0.0, 0, 0, 0
    carry = np.zeros(0, dtype=np.float32)
    total = info.frames or 1
    for block in sf.blocks(str(wav), blocksize=BLOCK, dtype="float32", always_2d=False):
        x = np.concatenate([carry, block])
        usable = (len(x) // F100) * F100
        carry = x[usable:]
        x = x[:usable]
        if usable == 0:
            continue
        n += usable
        ax = np.abs(x)
        peak = max(peak, float(ax.max()))
        hot = ax >= 0.999
        clips += int(hot.sum())
        if hot.any():  # count runs of >=3 consecutive clipped samples
            d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            clip_events += int(((ends - starts) >= 3).sum())
        ms10.append((x.reshape(-1, F10) ** 2).mean(axis=1))
        y, z1 = lfilter(b1, a1, x, zi=z1)
        y, z2 = lfilter(b2, a2, y, zi=z2)
        kw100.append((y.reshape(-1, F100) ** 2).mean(axis=1))
        progress(min(n / total, 1.0), f"Audio pass 1: {n / total:.0%}")
    return Envelope(
        np.concatenate(ms10) if ms10 else np.zeros(1),
        np.concatenate(kw100) if kw100 else np.zeros(1),
        peak,
        clips,
        clip_events,
        n,
    )


def find_silences(env: Envelope, min_dur: float = 0.5) -> list[tuple[float, float]]:
    db = env.db10
    live = db[db > DIGITAL_SILENCE_DB]
    floor = np.percentile(live, 20) if live.size else -60.0
    thr = min(floor + 8.0, -35.0)
    quiet = np.concatenate([[False], db < thr, [False]])
    d = np.diff(quiet.astype(np.int8))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s * 0.01, e * 0.01) for s, e in zip(starts, ends) if (e - s) * 0.01 >= min_dur]


# ---- loudness (BS.1770-4 gating) ----------------------------------------------
def _gated_loudness(kw100: np.ndarray) -> float:
    if kw100.size < 4:
        return float("nan")
    blocks = np.convolve(kw100, np.ones(4) / 4, mode="valid")  # 400 ms, 75 % overlap
    lk = -0.691 + _db(blocks)
    g = blocks[lk > -70]
    if g.size == 0:
        return float("nan")
    rel = -0.691 + _db(g.mean()) - 10
    g2 = g[(-0.691 + _db(g)) > rel]
    return float(-0.691 + _db(g2.mean())) if g2.size else float("nan")


def _loudness_range(kw100: np.ndarray) -> float:
    if kw100.size < 30:
        return float("nan")
    st = np.convolve(kw100, np.ones(30) / 30, mode="valid")[::10]  # 3 s windows, 1 s hop
    l = -0.691 + _db(st)
    l = l[l > -70]
    if l.size < 2:
        return float("nan")
    l = l[l > (-0.691 + _db(st[(-0.691 + _db(st)) > -70].mean()) - 20)]
    return float(np.percentile(l, 95) - np.percentile(l, 10)) if l.size >= 2 else float("nan")


# ---- helpers ---------------------------------------------------------------------
def _mask(intervals: list[tuple[float, float]], n: int, hop: float, pad: float = 0.0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        a, b = int(max(0, (s - pad) / hop)), int(min(n, np.ceil((e + pad) / hop)))
        if b > a:
            m[a:b] = True
    return m


def _interp(x: float, xs, ys) -> float:
    return float(np.interp(x, xs, ys))


def _metric(name: str, score: float, value: str, explanation: str, **extra) -> dict:
    return {"name": name, "score": round(float(np.clip(score, 0, 100)), 1), "value": value, "explanation": explanation, **extra}


# ---- reverb ---------------------------------------------------------------------------
def _reverb_rt(db10: np.ndarray, speech_mask10: np.ndarray) -> tuple[float, int]:
    """Blind RT60 estimate from free decays at speech offsets.

    A decay can never be faster than the room allows, so the steepest decays
    observed at word endings approximate the room's own decay rate.
    """
    sm = np.convolve(db10, np.ones(3) / 3, mode="same")
    ref = np.median(sm[speech_mask10]) if speech_mask10.any() else np.median(sm)
    rts = []
    i, n = 1, len(sm)
    while i < n - 40:
        if speech_mask10[i] and sm[i] >= ref - 3 and sm[i] >= sm[i - 1] and sm[i + 1] < sm[i]:
            j = i
            while j + 1 < n and sm[j + 1] < sm[j] + 0.5 and j - i < 150:
                j += 1
            drop = sm[i] - sm[j]
            if drop >= 25 and (j - i) >= 5:
                seg = sm[i : j + 1]
                lo = np.argmax(seg <= sm[i] - 5)
                hi = np.argmax(seg <= sm[i] - 25)
                if hi > lo + 2:
                    t = np.arange(lo, hi + 1) * 0.01
                    slope = np.polyfit(t, seg[lo : hi + 1], 1)[0]
                    if slope < 0:
                        rts.append(-60.0 / slope)
            i = j + 1
        else:
            i += 1
    if len(rts) < 10:
        return float("nan"), len(rts)
    return float(np.percentile(rts, 20)), len(rts)


def analyze(wav: Path, env: Envelope, tr: Transcript, progress: Progress) -> dict:
    n100 = env.kw100.size
    n10 = env.ms10.size
    dur = env.n_samples / SR
    speech_iv = [(s.start, s.end) for s in tr.segments]
    teacher_iv = [(s.start, s.end) for s in tr.segments if s.role == "TEACHER"]
    speech100 = _mask(speech_iv, n100, 0.1)
    teacher100 = _mask(teacher_iv, n100, 0.1)
    nonspeech100 = ~_mask(speech_iv, n100, 0.1, pad=0.4)
    speech10 = _mask(speech_iv, n10, 0.01)
    db100 = _db(np.add.reduceat(env.ms10, np.arange(0, n10, 10))[:n100] / 10) if n10 else np.zeros(n100)
    digital100 = db100 <= DIGITAL_SILENCE_DB

    # ---- pass 2: spectra (n_fft 4096 -> 3.9 Hz bins for hum detection) -------
    nfft = 4096
    win = np.hanning(nfft).astype(np.float32)
    spec_noise = np.zeros(nfft // 2 + 1)
    spec_speech = np.zeros(nfft // 2 + 1)
    n_noise = n_speech = 0
    rolloffs = []
    pos = 0
    carry = np.zeros(0, dtype=np.float32)
    total = max(env.n_samples, 1)
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    for block in sf.blocks(str(wav), blocksize=BLOCK, dtype="float32", always_2d=False):
        x = np.concatenate([carry, block])
        k = len(x) // nfft
        carry = x[k * nfft :]
        if k == 0:
            continue
        frames = x[: k * nfft].reshape(k, nfft)
        centers = (pos + np.arange(k) * nfft + nfft // 2) / SR
        idx = np.minimum((centers / 0.1).astype(int), n100 - 1)
        P = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
        is_speech = teacher100[idx] if teacher100.any() else speech100[idx]
        is_noise = nonspeech100[idx] & ~digital100[idx]
        if is_noise.any():
            spec_noise += P[is_noise].sum(axis=0)
            n_noise += int(is_noise.sum())
        if is_speech.any():
            Ps = P[is_speech]
            spec_speech += Ps.sum(axis=0)
            n_speech += int(is_speech.sum())
            cs = np.cumsum(Ps, axis=1)
            ro = (cs < 0.99 * cs[:, -1:]).sum(axis=1)
            rolloffs.extend(freqs[np.minimum(ro, len(freqs) - 1)][::4].tolist())
        pos += k * nfft
        progress(min(pos / total, 1.0), f"Audio pass 2: {pos / total:.0%}")

    metrics: dict[str, dict] = {}

    # 1. Loudness ------------------------------------------------------------------
    lufs_all = _gated_loudness(env.kw100)
    lufs_teacher = _gated_loudness(env.kw100[teacher100]) if teacher100.sum() > 20 else lufs_all
    lra = _loudness_range(env.kw100)
    L = lufs_teacher if np.isfinite(lufs_teacher) else lufs_all
    if np.isfinite(L):
        dev = max(0.0, -24 - L, L - (-14))
        verdict = "within the comfortable speech range (-24 to -14 LUFS)" if dev == 0 else (
            "too quiet - listeners must turn volume up" if L < -24 else "too loud / hot")
        metrics["loudness"] = _metric(
            "Volume level", 100 - 12 * dev, f"{L:.1f} LUFS (teacher speech)",
            f"Teacher speech loudness is {L:.1f} LUFS, {verdict}. Whole-recording loudness {lufs_all:.1f} LUFS"
            + (f", loudness range {lra:.1f} LU." if np.isfinite(lra) else "."),
            lufs_teacher=round(L, 1), lufs_integrated=round(lufs_all, 1) if np.isfinite(lufs_all) else None,
            loudness_range_lu=round(lra, 1) if np.isfinite(lra) else None,
        )
    else:
        metrics["loudness"] = _metric("Volume level", 0, "silent", "No audible programme loudness was measured.")

    # 2. Clipping ------------------------------------------------------------------
    ratio = env.clip_samples / max(env.n_samples, 1)
    peak_db = 20 * np.log10(max(env.peak, 1e-9))
    cscore = _interp(np.log10(ratio + 1e-9), [-6, -5, -4, -3, -2], [100, 100, 80, 50, 10])
    metrics["clipping"] = _metric(
        "Clipping / distortion", cscore, f"{ratio * 100:.4f}% samples, {env.clip_events} events",
        f"Peak level {peak_db:.1f} dBFS. {env.clip_events} clipping event(s) "
        + ("- no audible distortion expected." if cscore >= 95 else "- the loudest moments are likely distorted."),
        peak_dbfs=round(peak_db, 1), clip_events=env.clip_events,
    )

    # 3. Background noise / SNR ------------------------------------------------------
    live_nonspeech = nonspeech100 & ~digital100
    speech_lvl = np.median(db100[teacher100 & ~digital100]) if (teacher100 & ~digital100).sum() > 20 else (
        np.median(db100[speech100 & ~digital100]) if (speech100 & ~digital100).any() else np.nan)
    gated_share = digital100[nonspeech100].mean() if nonspeech100.any() else 0.0
    if live_nonspeech.sum() >= 300:  # >= 30 s of real background
        floor = float(np.median(db100[live_nonspeech]))
        method = "measured in pauses between speakers"
    else:
        inner = speech100 & ~digital100
        floor = float(np.percentile(db100[inner], 5)) if inner.any() else np.nan
        method = "estimated from pauses inside speech (the platform mutes silence)"
    snr = speech_lvl - floor if np.isfinite(speech_lvl) and np.isfinite(floor) else np.nan
    if n_noise > 10:
        S = spec_noise / n_noise
        band = (freqs > 100) & (freqs < 7000)
        flat = float(np.exp(np.mean(np.log(S[band] + EPS))) / (np.mean(S[band]) + EPS))
    else:
        flat = np.nan
    noise_type = ""
    if np.isfinite(flat) and np.isfinite(snr) and snr < 25:
        noise_type = " Background is broadband (fan/AC/hiss-like)." if flat > 0.3 else " Background has tonal/structured components."
    if np.isfinite(snr):
        nscore = _interp(snr, [5, 10, 15, 20, 30], [10, 35, 60, 80, 100])
        verdict = "clean" if snr >= 25 else "acceptable" if snr >= 15 else "noisy - background competes with the voice"
        metrics["noise"] = _metric(
            "Background noise (SNR)", nscore, f"SNR ≈ {snr:.0f} dB, floor {floor:.0f} dBFS",
            f"Speech sits about {snr:.0f} dB above the background ({verdict}); noise floor {method}."
            + noise_type + (f" {gated_share:.0%} of the gaps are digitally muted (noise suppression)." if gated_share > 0.3 else ""),
            snr_db=round(float(snr), 1), noise_floor_dbfs=round(float(floor), 1), spectral_flatness=None if not np.isfinite(flat) else round(flat, 3),
        )
    else:
        metrics["noise"] = _metric("Background noise (SNR)", 50, "n/a", "Not enough speech to estimate SNR.")

    # 4. Mains hum -------------------------------------------------------------------
    S = spec_noise / n_noise if n_noise > 10 else (spec_speech / n_speech if n_speech else None)
    hum_db, hum_f = 0.0, None
    if S is not None:
        df = freqs[1]
        for f0 in (50.0, 60.0):
            proms = []
            for h in range(1, 6):
                f = f0 * h
                k = int(round(f / df))
                peak_p = S[max(k - 1, 0) : k + 2].max()
                nb = np.r_[S[max(k - 6, 0) : max(k - 2, 0)], S[k + 3 : k + 7]]
                proms.append(_db(peak_p) - _db(np.median(nb) if nb.size else peak_p))
            p = float(np.mean(sorted(proms)[-3:]))
            if p > hum_db:
                hum_db, hum_f = p, f0
    hscore = _interp(hum_db, [6, 12, 20], [100, 60, 10])
    metrics["hum"] = _metric(
        "Electrical hum", hscore, f"{hum_db:.1f} dB peak at {hum_f:.0f} Hz series" if hum_f else "none",
        "No mains hum detected." if hum_db < 6 else f"Hum at {hum_f:.0f} Hz and harmonics stands {hum_db:.0f} dB above surrounding noise (ground loop / cable).",
        hum_prominence_db=round(hum_db, 1),
    )

    # 5. Sudden interruptions (keyboard clacks, bangs, bursts) -----------------------
    db10 = env.db10
    k = 50
    med = np.convolve(db10, np.ones(k) / k, mode="same")
    jump = (db10 - med > 18) & (db10 > -45)
    nonspeech10 = ~_mask(speech_iv, n10, 0.01, pad=0.3)
    ev = jump & nonspeech10
    d = np.diff(np.concatenate([[0], ev.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    merged = [s for i, s in enumerate(starts) if i == 0 or s - starts[i - 1] > 20]
    loud_ref = np.median(db100[teacher100]) if teacher100.any() else np.median(db100)
    bursts = np.where(db100 > loud_ref + 15)[0]
    burst_events = [b for i, b in enumerate(bursts) if i == 0 or b - bursts[i - 1] > 10]
    per_min = (len(merged) + len(burst_events)) / max(dur / 60, 1e-6)
    examples = sorted([s * 0.01 for s in merged] + [b * 0.1 for b in burst_events])[:6]
    metrics["interruptions"] = _metric(
        "Sudden noises / interruptions", _interp(per_min, [0.5, 2, 5, 10], [100, 70, 40, 10]),
        f"{per_min:.1f} / min",
        f"{len(merged)} transient noise(s) outside speech (keyboard, clicks, bangs) and {len(burst_events)} sudden loud burst(s)."
        + (" Examples at " + ", ".join(fmt_ts(t) for t in examples) + "." if examples else ""),
        events=len(merged) + len(burst_events), timestamps=[fmt_ts(t) for t in examples],
    )

    # 6. Reverb -------------------------------------------------------------------------
    rt, n_dec = _reverb_rt(db10, speech10)
    if np.isfinite(rt):
        rscore = _interp(rt, [0.35, 0.5, 0.7, 1.0, 1.5], [100, 80, 55, 25, 5])
        verdict = "dry, close-mic sound" if rt < 0.4 else "some room echo" if rt < 0.7 else "strong echo - tiring to listen to"
        metrics["reverb"] = _metric(
            "Room echo (reverb)", rscore, f"RT60 ≈ {rt:.2f} s",
            f"Estimated reverberation time {rt:.2f} s from {n_dec} word endings: {verdict}.", rt60_s=round(rt, 2),
        )
    else:
        metrics["reverb"] = _metric(
            "Room echo (reverb)", 85, "not measurable",
            f"Only {n_dec} clean word endings found (noise suppression usually truncates echo tails); assumed low reverb.",
        )

    # 7. Dropouts (digital silence inside someone's speech) -------------------------
    zero10 = db10 <= DIGITAL_SILENCE_DB
    inside = _mask([(s + 0.3, e - 0.3) for s, e in teacher_iv if e - s > 1.0], n10, 0.01)
    dz = np.diff(np.concatenate([[0], (zero10 & inside).astype(np.int8), [0]]))
    zs, ze = np.where(dz == 1)[0], np.where(dz == -1)[0]
    drops = [(s, e) for s, e in zip(zs, ze) if 5 <= (e - s) <= 150]
    teach_min = max(sum(e - s for s, e in teacher_iv) / 60, 1e-6)
    per10 = len(drops) / teach_min * 10
    metrics["dropouts"] = _metric(
        "Audio dropouts", _interp(per10, [0, 1, 3, 6, 10], [100, 85, 60, 30, 10]), f"{len(drops)} dropout(s)",
        "No mid-sentence dropouts." if not drops else
        f"{len(drops)} short gap(s) of digital silence while the teacher was speaking (network/packet loss), e.g. at "
        + ", ".join(fmt_ts(s * 0.01) for s, _ in drops[:5]) + ".",
        timestamps=[fmt_ts(s * 0.01) for s, _ in drops[:10]],
    )

    # 8. Consistency of the teacher's level ----------------------------------------
    per_seg = []
    for s in tr.segments:
        if s.role == "TEACHER" and s.duration >= 1.5:
            a, b = int(s.start / 0.1), int(s.end / 0.1)
            seg = env.kw100[a:b]
            if seg.size:
                per_seg.append(-0.691 + _db(seg.mean()))
    if len(per_seg) >= 5:
        sd = float(np.std(per_seg))
        metrics["consistency"] = _metric(
            "Voice level consistency", _interp(sd, [3, 5, 8, 12], [100, 80, 50, 15]), f"±{sd:.1f} dB",
            f"The teacher's level varies by ±{sd:.1f} dB between turns "
            + ("(steady)." if sd < 4 else "(noticeable volume jumps - mic distance or auto-gain)."),
        )
    else:
        metrics["consistency"] = _metric("Voice level consistency", 80, "n/a", "Too few long teacher turns to measure.")

    bw = float(np.median(rolloffs)) if rolloffs else float("nan")
    info = {
        "duration_s": round(dur, 1),
        "speech_bandwidth_hz": None if not np.isfinite(bw) else round(bw),
        "bandwidth_note": "" if not np.isfinite(bw) else (
            "Narrowband/muffled voice (phone-quality)." if bw < 3500 else "Full speech bandwidth."),
    }
    w = config.AUDIO_WEIGHTS
    score = sum(metrics[k]["score"] * w[k] for k in w) / sum(w.values())
    for k in metrics:
        metrics[k]["weight"] = w[k]
    return {"score": round(score, 1), "metrics": metrics, "info": info}
