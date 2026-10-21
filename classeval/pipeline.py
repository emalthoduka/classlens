"""End-to-end orchestration with per-stage progress, timing and caching."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import audio_eval, cache, config, ingest, llm_eval, media, scoring, transcribe, video_eval
from .errors import UserFacingError
from .transcript import Transcript, build_transcript, class_dynamics, parse_vtt

STAGES = [
    ("ingest", "Load recording"),
    ("audio", "Extract audio"),
    ("transcript", "Transcribe & identify teacher"),
    ("audio_eval", "Audio quality analysis"),
    ("video_eval", "Video quality & branded background"),
    ("llm", "Teaching quality evaluation"),
    ("score", "Scoring & report"),
]


@dataclass
class StageState:
    label: str
    status: str = "pending"  # pending | running | done | cached | skipped | error
    frac: float = 0.0
    desc: str = ""
    started: float | None = None
    elapsed: float | None = None


@dataclass
class Tracker:
    stages: dict[str, StageState] = field(default_factory=lambda: {k: StageState(v) for k, v in STAGES})
    lock: threading.Lock = field(default_factory=threading.Lock)
    t0: float = field(default_factory=time.time)

    def start(self, key: str, desc: str = "") -> None:
        with self.lock:
            s = self.stages[key]
            s.status, s.started, s.desc, s.frac = "running", time.time(), desc, 0.0

    def update(self, key: str, frac: float, desc: str = "") -> None:
        with self.lock:
            s = self.stages[key]
            s.frac = max(0.0, min(1.0, frac))
            if desc:
                s.desc = desc

    def finish(self, key: str, status: str = "done", desc: str = "") -> None:
        with self.lock:
            s = self.stages[key]
            s.status, s.frac = status, 1.0
            s.elapsed = (time.time() - s.started) if s.started else 0.0
            if desc:
                s.desc = desc

    def fail(self, key: str, msg: str) -> None:
        with self.lock:
            s = self.stages[key]
            s.status, s.desc = "error", msg
            s.elapsed = (time.time() - s.started) if s.started else 0.0

    def snapshot(self) -> list[dict]:
        with self.lock:
            now = time.time()
            return [
                {
                    "key": k,
                    "label": s.label,
                    "status": s.status,
                    "frac": s.frac,
                    "desc": s.desc,
                    "elapsed": s.elapsed if s.elapsed is not None else (now - s.started if s.started else None),
                }
                for k, s in self.stages.items()
            ]

    def overall(self) -> float:
        snap = self.snapshot()
        return sum(1.0 if s["status"] in ("done", "cached", "skipped") else s["frac"] for s in snap) / len(snap)

    def cb(self, key: str):
        return lambda frac, desc="": self.update(key, frac, desc)


@dataclass
class Job:
    video_path: str | None = None
    url: str | None = None
    vtt_path: str | None = None
    teacher_name: str = ""
    delete_upload: bool = True


def run(job: Job, tr_: Tracker) -> dict:
    cache.clean_stale_work_dirs()
    work = config.WORK_DIR / uuid.uuid4().hex[:12]
    work.mkdir(parents=True, exist_ok=True)
    current = "ingest"
    try:
        # ---- 1. ingest ----------------------------------------------------------
        tr_.start("ingest", "Preparing…")
        video: Path | None = None
        if job.url:
            vkey = cache.text_hash(ingest.source_key_for_url(job.url))
        elif job.video_path:
            video = Path(job.video_path)
            tr_.update("ingest", 0.2, "Fingerprinting upload…")
            vkey = cache.partial_file_hash(video)
        else:
            raise UserFacingError("Please upload a recording or paste a Google Drive / video link.")

        vtt_text = Path(job.vtt_path).read_text(encoding="utf-8-sig", errors="replace") if job.vtt_path else ""
        tkey = cache.text_hash("vtt" if vtt_text else config.TRANSCRIBE_MODEL, vtt_text, job.teacher_name.strip().lower())
        sc = cache.StageCache(vkey)
        cached_tr = sc.get(f"transcript_{tkey}")
        cached_audio = sc.get(f"audio_{tkey}")
        cached_video = sc.get(f"video_{tkey}_{config.VIDEO_EVAL_VERSION}")
        need_media = not (cached_tr and cached_audio and cached_video)

        if need_media and video is None:
            video = ingest.download_from_url(job.url, work / "dl", tr_.cb("ingest"))
        if need_media:
            tr_.update("ingest", 0.95, "Reading metadata…")
            info = media.probe(video)
            if not info["has_audio"]:
                raise UserFacingError("This recording has no audio track, so the class cannot be evaluated.")
            sc.put("probe", info)
            mb = info["size_bytes"] / 1024**2
            tr_.finish("ingest", desc=f"{info['duration'] / 60:.1f} min, {mb:,.0f} MB")
        else:
            info = sc.get("probe") or {}
            tr_.finish("ingest", "cached", "Results found in cache - media processing skipped")

        # ---- 2. audio extraction + envelope -------------------------------------------
        current = "audio"
        env = wav = mp3 = None
        if need_media:
            tr_.start("audio", "Decoding audio track…")
            wav, mp3 = media.extract_audio(video, work, info["duration"], lambda f, d: tr_.update("audio", f * 0.8, d))
            env = audio_eval.envelope(wav, lambda f, d: tr_.update("audio", 0.8 + 0.2 * f, d))
            tr_.finish("audio", desc=f"{env.n_samples / 16000 / 60:.1f} min of audio")
        else:
            tr_.finish("audio", "cached")

        # ---- 3. transcript ----------------------------------------------------------
        current = "transcript"
        if cached_tr:
            transcript = Transcript.from_dict(cached_tr)
            tr_.finish("transcript", "cached", f"{len(transcript.segments)} segments")
        else:
            tr_.start("transcript")
            if vtt_text:
                tr_.update("transcript", 0.3, "Parsing uploaded transcript…")
                segs = parse_vtt(job.vtt_path)
                if not segs:
                    raise UserFacingError("The uploaded transcript has no cues. Is it a valid .vtt file?")
                transcript = build_transcript(segs, "vtt", job.teacher_name)
            else:
                silences = audio_eval.find_silences(env)
                transcript = transcribe.transcribe(mp3, wav, info["duration"], silences, work, tr_.cb("transcript"))
            sc.put(f"transcript_{tkey}", transcript.to_dict())
            tr_.finish(
                "transcript",
                desc=f"{len(transcript.segments)} segments · teacher = '{transcript.teacher_speaker}' ({transcript.mapping_confidence} confidence)",
            )

        # ---- 4-6: LLM in background while audio/video run locally -----------------
        lkey = cache.text_hash(config.LLM_MODEL, config.LLM_REASONING_EFFORT, config.PROMPT_VERSION, tkey)
        llm_cached = sc.get(f"llm_{lkey}")
        pool = ThreadPoolExecutor(max_workers=1)
        llm_future = None
        if llm_cached:
            tr_.finish("llm", "cached")
        else:
            tr_.start("llm", f"Waiting for {config.LLM_MODEL}…")
            llm_future = pool.submit(llm_eval.evaluate, transcript)

        current = "audio_eval"
        if cached_audio:
            audio_res = cached_audio
            tr_.finish("audio_eval", "cached")
        else:
            tr_.start("audio_eval")
            audio_res = audio_eval.analyze(wav, env, transcript, tr_.cb("audio_eval"))
            sc.put(f"audio_{tkey}", audio_res)
            tr_.finish("audio_eval", desc=f"score {audio_res['score']:.0f}/100")

        current = "video_eval"
        if cached_video:
            video_res = None if cached_video.get("skipped") else cached_video
            tr_.finish("video_eval", "cached")
        elif not info.get("has_video"):
            video_res = None
            sc.put(f"video_{tkey}_{config.VIDEO_EVAL_VERSION}", {"skipped": True})
            tr_.start("video_eval")
            tr_.finish("video_eval", "skipped", "No video stream (audio-only file)")
        else:
            tr_.start("video_eval")
            video_res = video_eval.analyze(video, info, transcript, work, sc.path(f"thumbs_{tkey}"), tr_.cb("video_eval"))
            sc.put(f"video_{tkey}_{config.VIDEO_EVAL_VERSION}", video_res)
            tr_.finish("video_eval", desc=f"score {video_res['score']:.0f}/100")

        # The heavy media is no longer needed: free disk immediately.
        if video is not None and (job.url or job.delete_upload):
            video.unlink(missing_ok=True)
        for p in (wav, mp3):
            if p is not None:
                Path(p).unlink(missing_ok=True)

        current = "llm"
        if llm_future is not None:
            while not llm_future.done():
                el = time.time() - (tr_.stages["llm"].started or time.time())
                tr_.update("llm", min(0.95, el / 90), f"{config.LLM_MODEL} is reading the transcript… {el:.0f}s")
                time.sleep(0.5)
            llm_res = llm_future.result()
            sc.put(f"llm_{lkey}", llm_res)
            ev = llm_res.get("evidence_check", {})
            tr_.finish("llm", desc=f"{ev.get('verified', 0)}/{ev.get('checked', 0)} evidence quotes verified")
        else:
            llm_res = llm_cached
        pool.shutdown(wait=False)

        # ---- 7. scoring ---------------------------------------------------------------
        current = "score"
        tr_.start("score")
        teaching = scoring.teaching_score(llm_res)
        result = scoring.final(teaching, audio_res, video_res)
        report = {
            "key": vkey,
            "final": result,
            "teaching": teaching,
            "llm": {k: llm_res.get(k) for k in ("lesson_topic", "summary", "strengths", "improvements",
                                                "new_student_present", "video_shown", "evidence_check", "_meta")},
            "audio": audio_res,
            "video": video_res,
            "transcript": {
                "source": transcript.source,
                "teacher_speaker": transcript.teacher_speaker,
                "mapping_confidence": transcript.mapping_confidence,
                "mapping_note": transcript.mapping_note,
                "speaker_stats": transcript.speaker_stats,
                "dynamics": class_dynamics(transcript),
                "segments": [
                    {"start": s.start, "end": s.end, "role": s.role, "speaker": s.speaker, "text": s.text}
                    for s in transcript.segments
                ],
            },
            "media": info,
            "config": {"llm_model": config.LLM_MODEL, "transcribe_model": config.TRANSCRIBE_MODEL,
                       "prompt_version": config.PROMPT_VERSION},
            "timings": {s["key"]: s["elapsed"] for s in tr_.snapshot()},
            "total_seconds": round(time.time() - tr_.t0, 1),
        }
        sc.put(f"report_{lkey}", report)
        tr_.finish("score", desc=f"{result['label']} · {result['score']:.1f}/100")
        cache.evict()
        return report
    except UserFacingError as e:
        tr_.fail(current, str(e))
        raise
    except Exception as e:
        tr_.fail(current, f"Unexpected error: {e.__class__.__name__}")
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
