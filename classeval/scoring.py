"""Turn pillar results into the final 0-100 score and quality class."""
from __future__ import annotations

from . import config


def teaching_score(llm: dict) -> dict:
    """Weighted mean of applicable criteria (0-10 -> 0-100) minus incident penalties."""
    rows = []
    num = den = 0.0
    for key, (name, weight, kind) in config.TEACHING_CRITERIA.items():
        c = llm["criteria"].get(key) or {"applicable": False, "score": 0, "rationale": "", "evidence": []}
        applicable = bool(c.get("applicable", True))
        if kind == "core":
            applicable = True  # core practices can't be opted out of; absence scores low
        elif kind == "bonus" and c.get("score", 0) == 0:
            applicable = False
        if applicable:
            num += weight * c["score"] * 10
            den += weight
        rows.append({"key": key, "name": name, "weight": weight, "kind": kind, "applicable": applicable, **c})
    base = num / den if den else 0.0

    incidents = [i for i in llm.get("negative_incidents", []) if i.get("status") != "not_teacher"]
    penalty = 0.0
    for inc in incidents:
        p = config.INCIDENT_PENALTY.get(inc.get("severity", "minor"), 2.0)
        if not inc.get("verified", False):
            p *= 0.5  # unverifiable quotes count half
        inc["penalty"] = p
        penalty += p
    penalty = min(penalty, config.INCIDENT_PENALTY_CAP)
    severe = any(i.get("severity") == "severe" and i.get("verified") for i in incidents)
    return {
        "score": round(max(0.0, base - penalty), 1),
        "base_score": round(base, 1),
        "penalty": round(penalty, 1),
        "criteria": rows,
        "incidents": incidents,
        "severe_incident": severe,
    }


def classify(score: float) -> str:
    for label, lo in config.CLASS_THRESHOLDS:
        if score >= lo:
            return label
    return "Poor"


_ORDER = [c for c, _ in config.CLASS_THRESHOLDS]  # best -> worst


def final(teaching: dict, audio: dict | None, video: dict | None) -> dict:
    parts = {"teaching": teaching["score"], "audio": audio and audio["score"], "video": video and video["score"]}
    weights = {k: w for k, w in config.PILLAR_WEIGHTS.items() if parts[k] is not None}
    total_w = sum(weights.values())
    score = sum(parts[k] * w for k, w in weights.items()) / total_w
    label = classify(score)
    notes = []
    if len(weights) < len(config.PILLAR_WEIGHTS):
        missing = [k for k in config.PILLAR_WEIGHTS if k not in weights]
        notes.append(f"{', '.join(missing)} could not be evaluated; remaining weights were rescaled.")
    if teaching.get("severe_incident") and _ORDER.index(label) < _ORDER.index(config.SEVERE_INCIDENT_CAP):
        notes.append(f"Capped at '{config.SEVERE_INCIDENT_CAP}' because of a severe teacher incident (see incidents).")
        label = config.SEVERE_INCIDENT_CAP
    return {
        "score": round(score, 1),
        "label": label,
        "weights": {k: round(w / total_w, 3) for k, w in weights.items()},
        "contributions": {k: round(parts[k] * w / total_w, 1) for k, w in weights.items()},
        "thresholds": config.CLASS_THRESHOLDS,
        "notes": notes,
    }
