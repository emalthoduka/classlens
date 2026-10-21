"""A schema-shaped stand-in for the LLM so the pipeline can be tested offline."""
from __future__ import annotations

from classeval import config
from classeval.llm_eval import verify_evidence
from classeval.transcript import Transcript, fmt_ts


def fake_evaluate(tr: Transcript) -> dict:
    t0 = tr.teacher_segments()[0]
    ev = {"timestamp": fmt_ts(t0.start), "quote": " ".join(t0.text.split()[:5]), "observation": "test", "polarity": "positive"}
    crit = {
        k: {"applicable": kind != "bonus", "score": 7 if kind != "bonus" else 0, "rationale": f"Synthetic rationale for {k}.",
            "evidence": [dict(ev)]}
        for k, (_, _, kind) in config.TEACHING_CRITERIA.items()
    }
    res = {
        "lesson_topic": "Back rank mate",
        "new_student_present": False,
        "video_shown": False,
        "criteria": crit,
        "negative_incidents": [{"timestamp": fmt_ts(t0.start), "quote": "made up words never said", "category": "impatience",
                                "severity": "minor", "explanation": "test incident"}],
        "summary": "Synthetic summary.",
        "strengths": ["Clear explanation"],
        "improvements": ["Add a recap"],
        "_meta": {"model": "fake", "prompt_tokens": 0, "completion_tokens": 0},
    }
    return verify_evidence(res, tr)
