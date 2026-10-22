"""Teaching-quality evaluation with one structured-output LLM call."""
from __future__ import annotations

import json

from rapidfuzz import fuzz

from . import config
from .errors import UserFacingError, explain_openai_error
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .transcript import Segment, Transcript, class_dynamics, fmt_ts, parse_ts, render_for_llm

INCIDENT_CATEGORIES = [
    "frustration",
    "impatience",
    "raised_voice_or_shouting",
    "harsh_or_belittling",
    "sarcasm",
    "inappropriate_language",
    "ignoring_student",
    "other",
]


def _str(desc: str = "") -> dict:
    return {"type": "string", "description": desc} if desc else {"type": "string"}


def build_schema() -> dict:
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "timestamp": _str("mm:ss or h:mm:ss copied from the transcript line"),
            "quote": _str("short verbatim quote from that line"),
            "observation": _str(),
            "polarity": {"type": "string", "enum": ["positive", "negative"]},
        },
        "required": ["timestamp", "quote", "observation", "polarity"],
    }
    criterion = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "applicable": {"type": "boolean"},
            "score": {"type": "integer", "description": "0-10"},
            "rationale": _str(),
            "evidence": {"type": "array", "items": evidence},
        },
        "required": ["applicable", "score", "rationale", "evidence"],
    }
    incident = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "timestamp": _str(),
            "quote": _str(),
            "category": {"type": "string", "enum": INCIDENT_CATEGORIES},
            "severity": {"type": "string", "enum": ["minor", "moderate", "severe"]},
            "explanation": _str(),
        },
        "required": ["timestamp", "quote", "category", "severity", "explanation"],
    }
    crit_keys = list(config.TEACHING_CRITERIA)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "lesson_topic": _str(),
            "new_student_present": {"type": "boolean"},
            "video_shown": {"type": "boolean"},
            "criteria": {
                "type": "object",
                "additionalProperties": False,
                "properties": {k: criterion for k in crit_keys},
                "required": crit_keys,
            },
            "negative_incidents": {"type": "array", "items": incident},
            "summary": _str(),
            "strengths": {"type": "array", "items": _str()},
            "improvements": {"type": "array", "items": _str()},
        },
        "required": [
            "lesson_topic",
            "new_student_present",
            "video_shown",
            "criteria",
            "negative_incidents",
            "summary",
            "strengths",
            "improvements",
        ],
    }


# ---- Evidence verification ----------------------------------------------------
def _verify(ts: str, quote: str, segments: list[Segment], window: float = 25.0) -> tuple[str, bool, str]:
    """Check a cited (timestamp, quote) against the transcript.

    Returns (timestamp, verified, status) where status is 'verified', 'corrected'
    (quote found elsewhere -> timestamp fixed) or 'unverified'.
    """
    q = quote.strip().strip('"').lower()
    t = parse_ts(ts)
    if not q:
        return ts, False, "unverified"

    def score(s: Segment) -> float:
        return fuzz.partial_ratio(q, s.text.lower()) if len(q) <= len(s.text) + 5 else fuzz.ratio(q, s.text.lower())

    if t is not None:
        near = [s for s in segments if s.start - window <= t <= s.end + window]
        best = max(near, key=score, default=None)
        if best is not None and score(best) >= 75:
            return fmt_ts(best.start), True, "verified"
    best_all = max(segments, key=score, default=None)
    if best_all is not None and score(best_all) >= 85 and len(q) >= 12:
        return fmt_ts(best_all.start), True, "corrected"
    return ts, False, "unverified"


def verify_evidence(result: dict, tr: Transcript) -> dict:
    segs = tr.segments
    checked = ok = 0
    for crit in result["criteria"].values():
        crit["score"] = int(min(10, max(0, crit.get("score", 0))))
        for ev in crit["evidence"]:
            ev["timestamp"], ev["verified"], ev["status"] = _verify(ev["timestamp"], ev["quote"], segs)
            checked += 1
            ok += ev["verified"]
    for inc in result["negative_incidents"]:
        inc["timestamp"], inc["verified"], inc["status"] = _verify(inc["timestamp"], inc["quote"], segs)
        # Only teacher lines can be teacher incidents.
        t = parse_ts(inc["timestamp"])
        if inc["verified"] and t is not None:
            line = min(segs, key=lambda s: abs(s.start - t))
            if line.role != "TEACHER" and fuzz.partial_ratio(inc["quote"].lower(), line.text.lower()) >= 75:
                inc["verified"], inc["status"] = False, "not_teacher"
    result["evidence_check"] = {"checked": checked, "verified": ok}
    return result


def evaluate(tr: Transcript) -> dict:
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise UserFacingError("OPENAI_API_KEY is not configured. Add it as a secret in the app settings.")
    client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=4, timeout=600)
    stats = class_dynamics(tr)
    user = USER_TEMPLATE.format(stats=json.dumps(stats), transcript=render_for_llm(tr))
    kwargs: dict = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "class_evaluation", "strict": True, "schema": build_schema()},
        },
        "max_completion_tokens": 20000,
    }
    if config.LLM_MODEL.startswith(("gpt-5", "gpt-6", "o")):
        if config.LLM_REASONING_EFFORT:
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
    else:
        kwargs["temperature"] = 0.2
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        raise explain_openai_error(e, "Teaching evaluation") from e

    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None):
        raise UserFacingError(f"The model declined to evaluate this transcript: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise UserFacingError("The evaluation was cut off (output too long). Try again or use a larger model.")
    try:
        result = json.loads(choice.message.content or "")
    except json.JSONDecodeError as e:
        raise UserFacingError("The evaluation model returned malformed output. Please retry.") from e

    usage = getattr(resp, "usage", None)
    result["_meta"] = {
        "model": resp.model,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    return verify_evidence(result, tr)
