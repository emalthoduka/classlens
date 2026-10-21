"""Diarized transcription with OpenAI, chunked for long recordings.

Speaker labels from the diarizer are only consistent *within* one request, so a
long class is processed as:

1. Transcribe chunk 0 and pick the teacher there (most talk time).
2. Cut a clean 3-10 s clip of the teacher's voice from chunk 0.
3. Transcribe the remaining chunks in parallel, passing that clip as a
   `known_speaker_references` sample named "TEACHER" so the teacher keeps one
   label across the whole class - no extra LLM call needed.
"""
from __future__ import annotations

import base64
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from . import config, media
from .errors import UserFacingError, explain_openai_error
from .transcript import TEACHER, Segment, Transcript, assign_roles, fmt_ts, pick_teacher, speaker_stats

log = logging.getLogger("classeval.transcribe")

Progress = Callable[[float, str], None]


def _client():
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise UserFacingError("OPENAI_API_KEY is not configured. Add it as a secret in the Space settings.")
    # Fail fast on a hung request instead of waiting many minutes per silent retry.
    return OpenAI(api_key=config.OPENAI_API_KEY, max_retries=2, timeout=config.TRANSCRIBE_TIMEOUT)


def _call(client, chunk: Path, ref_data_url: str | None) -> list[dict]:
    kwargs: dict = {
        "model": config.TRANSCRIBE_MODEL,
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    if config.TRANSCRIBE_LANGUAGE:
        kwargs["language"] = config.TRANSCRIBE_LANGUAGE
    if ref_data_url:
        kwargs["known_speaker_names"] = [TEACHER]
        kwargs["known_speaker_references"] = [ref_data_url]
    with chunk.open("rb") as f:
        resp = client.audio.transcriptions.create(file=f, **kwargs)
    segs = getattr(resp, "segments", None) or []
    return [
        {"start": float(s.start), "end": float(s.end), "speaker": str(s.speaker), "text": (s.text or "").strip()}
        for s in segs
        if (s.text or "").strip()
    ]


def _reference_clip(wav: Path, segs: list[Segment], teacher: str, work: Path) -> str | None:
    """Pick the teacher's longest 3-10 s turn and encode it as a data URL."""
    cands = [s for s in segs if s.speaker == teacher and s.duration >= 3.0]
    if not cands:
        return None
    best = max(cands, key=lambda s: min(s.duration, 10.0) + 0.01 * len(s.text))
    start = best.start + 0.15
    end = min(best.end - 0.1, start + 9.0)
    if end - start < 2.5:
        return None
    clip = media.cut_audio(wav, start, end, work / "teacher_ref.wav", ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
    return "data:audio/wav;base64," + base64.b64encode(clip.read_bytes()).decode()


def transcribe(
    api_audio: Path,
    analysis_wav: Path,
    duration: float,
    silences: list[tuple[float, float]],
    work: Path,
    progress: Progress,
) -> Transcript:
    client = _client()
    chunks = media.plan_chunks(duration, silences, config.TRANSCRIBE_CHUNK_SECONDS)
    n = len(chunks)
    files = []
    for i, (s, e) in enumerate(chunks):
        files.append(media.cut_audio(api_audio, s, e, work / f"chunk_{i:03d}.mp3"))
    log.info("Transcribing %.1f min of audio in %d chunk(s) of ~%ds, %d in parallel",
             duration / 60, n, config.TRANSCRIBE_CHUNK_SECONDS, config.TRANSCRIBE_PARALLELISM)
    t_start = time.time()

    def _timed(i: int, ref_url: str | None):
        t = time.time()
        log.info("chunk %d/%d started (%s-%s)", i + 1, n, fmt_ts(chunks[i][0]), fmt_ts(chunks[i][1]))
        raw = _call(client, files[i], ref_url)
        log.info("chunk %d/%d done in %.1fs (%d segments)", i + 1, n, time.time() - t, len(raw))
        return i, raw

    with ThreadPoolExecutor(max_workers=config.TRANSCRIBE_PARALLELISM) as ex:
        # Chunk 1 alone: the teacher's voice reference for all other chunks comes from it.
        fut0 = ex.submit(_timed, 0, None)
        while not fut0.done():
            wait([fut0], timeout=1.0)
            progress(0.02, f"Chunk 1/{n} (finding the teacher's voice)… {time.time() - t_start:.0f}s")
        try:
            _, raw0 = fut0.result()
        except UserFacingError:
            raise
        except Exception as e:  # noqa: BLE001 - translate any SDK error
            log.warning("chunk 1 failed: %r", e)
            raise explain_openai_error(e, "Transcription") from e

        off0 = chunks[0][0]
        seg0 = [Segment(r["start"] + off0, r["end"] + off0, f"c0:{r['speaker']}", r["text"]) for r in raw0]
        teacher0, conf0, _, _ = pick_teacher(seg0)
        ref = _reference_clip(analysis_wav, seg0, teacher0, work) if teacher0 else None
        log.info("teacher voice reference %s", "extracted" if ref else "NOT found - using talk-time per chunk")
        for s in seg0:
            if s.speaker == teacher0:
                s.speaker = TEACHER

        results: dict[int, list[Segment]] = {0: seg0}
        fallback_chunks: list[int] = []
        pending = {ex.submit(_timed, i, ref) for i in range(1, n)}
        while pending:
            finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    i, raw = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.warning("a chunk failed: %r", e)
                    for f in pending:
                        f.cancel()
                    raise explain_openai_error(e, "Transcription") from e
                off = chunks[i][0]
                segs = [
                    Segment(r["start"] + off, r["end"] + off, TEACHER if r["speaker"] == TEACHER else f"c{i}:{r['speaker']}", r["text"])
                    for r in raw
                ]
                if not any(s.speaker == TEACHER for s in segs) and segs:
                    # Reference matching failed for this chunk; fall back to talk-time heuristic.
                    t_i, c_i, _, _ = pick_teacher(segs)
                    if c_i in ("high", "medium"):
                        for s in segs:
                            if s.speaker == t_i:
                                s.speaker = TEACHER
                    fallback_chunks.append(i + 1)
                results[i] = segs
            done = len(results)
            progress(done / n, f"{done}/{n} chunks transcribed · {time.time() - t_start:.0f}s")
    log.info("transcription finished in %.1fs", time.time() - t_start)

    segments = [s for i in sorted(results) for s in results[i]]
    segments.sort(key=lambda s: s.start)
    assign_roles(segments, TEACHER)
    stats = speaker_stats(segments)
    total = sum(v["talk_seconds"] for v in stats.values()) or 1.0
    for v in stats.values():
        v["share"] = round(v["talk_seconds"] / total, 3)
    t_share = stats.get(TEACHER, {}).get("talk_seconds", 0.0) / total

    conf = conf0
    if fallback_chunks or ref is None:
        conf = "medium" if conf == "high" else conf
    note = (
        f"Diarized into {n} chunk(s). Teacher identified in chunk 1 by most talk time "
        f"({'voice reference used for later chunks' if ref else 'no clean voice reference found'}); "
        f"teacher holds {t_share:.0%} of total talk time."
    )
    if fallback_chunks:
        note += f" Voice matching failed for chunk(s) {fallback_chunks}; talk-time fallback used there."
    if not segments:
        raise UserFacingError("Transcription returned no speech. Does the recording contain audible speech?")
    return Transcript(segments, "openai-diarize", TEACHER, conf, note, stats)
