"""Offline unit tests (no OpenAI calls). Run: python -m pytest -q"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from classeval import config, llm_eval, media, scoring
from classeval.ingest import drive_file_id
from classeval.transcript import Segment, Transcript, build_transcript, parse_ts, parse_vtt, render_for_llm

ROOT = Path(__file__).resolve().parent.parent
REAL_VTT = ROOT / "samples" / "B491 Beginner -transcript-1.1.vtt"


def _segs(spec):
    return [Segment(s, e, who, txt) for s, e, who, txt in spec]


def test_teacher_is_most_talk_time_not_most_turns():
    # Student has more (short) turns; teacher has more talk time.
    segs = _segs([(0, 20, "T", "long explanation")] + [(21 + i, 21.5 + i, "kid", "yes") for i in range(15)])
    tr = build_transcript(segs, "vtt")
    assert tr.teacher_speaker == "T"
    assert all(s.role == ("TEACHER" if s.speaker == "T" else "STUDENT") for s in tr.segments)


def test_teacher_override():
    segs = _segs([(0, 20, "A", "x"), (20, 25, "B", "y")])
    assert build_transcript(segs, "vtt", "b").teacher_speaker == "B"


@pytest.mark.skipif(not REAL_VTT.exists(), reason="sample transcript not present")
def test_real_zoom_vtt():
    segs = parse_vtt(REAL_VTT)
    assert len(segs) > 900
    tr = build_transcript(segs, "vtt")
    assert tr.teacher_speaker == "Ayswarya Joseph"
    assert tr.mapping_confidence == "high"
    text = render_for_llm(tr)
    assert text.startswith("[00:21] TEACHER: Hello, Gautam!")
    assert len(text) < 60_000  # ~15k tokens: cheap for the LLM call


def test_parse_ts():
    assert parse_ts("01:05") == 65
    assert parse_ts("1:00:05.5") == 3605.5
    assert parse_ts("bad") is None


def test_drive_ids():
    assert drive_file_id("https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view?usp=sharing") == "1AbCdEfGhIjKlMn"
    assert drive_file_id("https://drive.google.com/open?id=1AbCdEfGhIjKlMn") == "1AbCdEfGhIjKlMn"
    assert drive_file_id("https://example.com/v.mp4") is None


def test_chunk_plan_prefers_silence():
    chunks = media.plan_chunks(1500, [(590, 596), (1190, 1192)], 600)
    assert chunks[0] == (0.0, 593.0)
    assert chunks[-1][1] == 1500
    assert all(e - s <= 750 for s, e in chunks)


def test_evidence_verification_and_correction():
    tr = Transcript(_segs([(10, 14, "T", "Okay, so today we learn the back rank mate"),
                           (300, 305, "T", "For homework solve five puzzles")]), "vtt")
    tr.segments[0].role = tr.segments[1].role = "TEACHER"
    assert llm_eval._verify("00:11", "today we learn the back rank", tr.segments)[2] == "verified"
    ts, ok, status = llm_eval._verify("00:12", "For homework solve five puzzles", tr.segments)
    assert ok and status == "corrected" and ts == "05:00"
    assert llm_eval._verify("00:12", "completely invented sentence here", tr.segments)[1] is False


def _llm(score=8, **over):
    crit = {k: {"applicable": kind != "bonus", "score": score if kind != "bonus" else 0, "rationale": "", "evidence": []}
            for k, (_, _, kind) in config.TEACHING_CRITERIA.items()}
    return {"criteria": crit, "negative_incidents": [], **over}


def test_scoring_and_thresholds():
    t = scoring.teaching_score(_llm(9))
    assert t["score"] == 90.0
    res = scoring.final(t, {"score": 80}, {"score": 70})
    assert res["score"] == pytest.approx(0.8 * 90 + 8 + 7)
    assert res["label"] == "Excellent"
    assert scoring.classify(69.9) == "Average" and scoring.classify(49.9) == "Poor"


def test_severe_incident_caps_class():
    llm = _llm(10, negative_incidents=[{"timestamp": "01:00", "quote": "x", "category": "raised_voice_or_shouting",
                                        "severity": "severe", "explanation": "", "verified": True}])
    t = scoring.teaching_score(llm)
    assert t["penalty"] == 10 and t["severe_incident"]
    res = scoring.final(t, {"score": 100}, {"score": 100})
    assert res["label"] == "Average" and res["notes"]


def test_missing_video_rescales_weights():
    t = scoring.teaching_score(_llm(5))
    res = scoring.final(t, {"score": 50}, None)
    assert res["score"] == 50 and set(res["weights"]) == {"teaching", "audio"}


def test_schema_is_strict():
    schema = llm_eval.build_schema()
    crit = schema["properties"]["criteria"]
    assert set(crit["required"]) == set(config.TEACHING_CRITERIA)
    assert schema["additionalProperties"] is False


def test_background_detector_synthetic():
    from classeval.background import detect
    from tests.synth import scenes

    results = [(name, detect(img).found == expected) for name, img, expected in scenes()]
    assert all(ok for _, ok in results), results


def test_k_weighting_1khz_gain_is_near_zero():
    from scipy.signal import lfilter

    from classeval.audio_eval import k_weighting

    (b1, a1), (b2, a2) = k_weighting()
    t = np.arange(16000) / 16000
    x = np.sin(2 * np.pi * 997 * t)
    y = lfilter(b2, a2, lfilter(b1, a1, x))
    gain_db = 10 * np.log10((y[4000:] ** 2).mean() / (x[4000:] ** 2).mean())
    assert abs(gain_db - 0.69) < 0.3  # BS.1770: +0.69 dB at 1 kHz, cancelled by the -0.691 offset


def test_openai_quota_error_is_friendly():
    import httpx
    import openai

    from classeval.errors import explain_openai_error

    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/x"))
    exc = openai.RateLimitError("quota", response=resp,
                                body={"error": {"code": "insufficient_quota", "message": "You exceeded your quota"}})
    msg = str(explain_openai_error(exc, "Transcription"))
    assert "run out of credit" in msg and "Billing" in msg
    auth = openai.AuthenticationError("bad", response=httpx.Response(401, request=resp.request), body=None)
    assert "OPENAI_API_KEY" in str(explain_openai_error(auth, "Teaching evaluation"))


def test_cache_handles_numpy_types(tmp_path):
    from classeval.cache import StageCache

    sc = StageCache("k", root=tmp_path)
    sc.put("x", {"b": np.bool_(False), "i": np.int64(3), "f": np.float32(1.5), "a": np.arange(2)})
    assert sc.get("x") == {"b": False, "i": 3, "f": 1.5, "a": [0, 1]}


def test_background_found_is_plain_bool():
    from classeval.background import detect
    from tests.synth import scenes

    for _, img, _ in scenes():
        assert type(detect(img).found) is bool


def test_brand_full_marks_after_a_few_sightings(monkeypatch, tmp_path):
    from classeval import video_eval
    from classeval.background import Match

    found = iter([True, False, True, True, False])
    monkeypatch.setattr(video_eval.background, "detect", lambda img: Match(next(found), 30))
    monkeypatch.setattr(video_eval.media, "grab_frames",
                        lambda video, times, out, prog, **kw: {t: _dummy_frame(tmp_path) for t in times})
    tr = build_transcript(_segs([(i * 60, i * 60 + 10, "T", "talk") for i in range(5)]), "vtt")
    monkeypatch.setattr(config, "TEACHER_FRAMES", 5)
    res = video_eval.analyze(Path("x.mp4"), {"duration": 300, "width": 1280, "height": 720, "avg_fps": 25},
                             tr, tmp_path, tmp_path / "th", lambda *a: None)
    m = res["metrics"]["brand_background"]
    assert m["score"] == 100.0 and m["value"].startswith("3/5")


def _dummy_frame(tmp_path):
    import cv2

    p = tmp_path / "f.jpg"
    if not p.exists():
        cv2.imwrite(str(p), np.full((72, 128, 3), 90, np.uint8))
    return p
