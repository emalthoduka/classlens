"""Transcript data model, Zoom VTT parsing and teacher/student role mapping."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

TEACHER = "TEACHER"
STUDENT = "STUDENT"


@dataclass
class Segment:
    start: float
    end: float
    speaker: str  # raw label from the source (diarizer letter or Zoom display name)
    text: str
    role: str = STUDENT

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    segments: list[Segment]
    source: str  # "openai-diarize" | "vtt"
    teacher_speaker: str = ""
    mapping_confidence: str = "high"  # high | medium | low
    mapping_note: str = ""
    speaker_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "segments": [asdict(s) for s in self.segments],
            "source": self.source,
            "teacher_speaker": self.teacher_speaker,
            "mapping_confidence": self.mapping_confidence,
            "mapping_note": self.mapping_note,
            "speaker_stats": self.speaker_stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segs = [Segment(**s) for s in d["segments"]]
        return cls(
            segments=segs,
            source=d["source"],
            teacher_speaker=d.get("teacher_speaker", ""),
            mapping_confidence=d.get("mapping_confidence", "high"),
            mapping_note=d.get("mapping_note", ""),
            speaker_stats=d.get("speaker_stats", {}),
        )

    @property
    def duration(self) -> float:
        return max((s.end for s in self.segments), default=0.0)

    def teacher_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.role == TEACHER]


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_ts(ts: str) -> float | None:
    """Parse 'mm:ss', 'h:mm:ss' or 'hh:mm:ss.mmm' into seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


# ---- Zoom VTT ----------------------------------------------------------------
_CUE_TIME = re.compile(r"(\d+:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d+:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})")
_SPEAKER = re.compile(r"^([^:]{1,60}):\s+(.*)$")


def parse_vtt(path: str | Path) -> list[Segment]:
    """Parse a WebVTT file. Zoom writes cues as 'Display Name: text'."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    segments: list[Segment] = []
    for block in re.split(r"\r?\n\s*\r?\n", text):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        for i, line in enumerate(lines):
            m = _CUE_TIME.search(line)
            if not m:
                continue
            body = " ".join(lines[i + 1 :]).strip()
            if not body:
                break
            sm = _SPEAKER.match(body)
            speaker, content = (sm.group(1).strip(), sm.group(2).strip()) if sm else ("UNKNOWN", body)
            # Strip simple VTT voice tags like <v Name>
            content = re.sub(r"</?v[^>]*>", "", content).strip()
            segments.append(Segment(parse_ts(m.group(1)) or 0.0, parse_ts(m.group(2)) or 0.0, speaker, content))
            break
    segments.sort(key=lambda s: s.start)
    return segments


# ---- Role mapping ------------------------------------------------------------
def speaker_stats(segments: list[Segment]) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: {"talk_seconds": 0.0, "turns": 0, "words": 0})
    for s in segments:
        st = stats[s.speaker]
        st["talk_seconds"] += s.duration
        st["turns"] += 1
        st["words"] += len(s.text.split())
    return {k: {**v, "talk_seconds": round(v["talk_seconds"], 1)} for k, v in stats.items()}


def pick_teacher(segments: list[Segment]) -> tuple[str, str, str, dict]:
    """Pick the teacher as the speaker with the most talk time (turns break ties).

    Talk time is more robust than turn count: children often produce many short
    turns ("yes", "okay") while the teacher produces fewer, longer ones.
    Returns (speaker, confidence, note, stats).
    """
    stats = speaker_stats(segments)
    if not stats:
        return "", "low", "Transcript is empty.", stats
    total = sum(v["talk_seconds"] for v in stats.values()) or 1.0
    ranked = sorted(stats.items(), key=lambda kv: (kv[1]["talk_seconds"], kv[1]["turns"]), reverse=True)
    teacher, top = ranked[0]
    share = top["talk_seconds"] / total
    runner_share = ranked[1][1]["talk_seconds"] / total if len(ranked) > 1 else 0.0
    if share >= 0.40 and share >= 1.5 * runner_share:
        conf = "high"
    elif share >= 0.30:
        conf = "medium"
    else:
        conf = "low"
    note = (
        f"'{teacher}' mapped to TEACHER with {share:.0%} of total talk time "
        f"({top['turns']} turns); next speaker has {runner_share:.0%}."
    )
    for v in stats.values():
        v["share"] = round(v["talk_seconds"] / total, 3)
    return teacher, conf, note, stats


def assign_roles(segments: list[Segment], teacher: str) -> None:
    for s in segments:
        s.role = TEACHER if s.speaker == teacher else STUDENT


def build_transcript(segments: list[Segment], source: str, teacher_override: str = "") -> Transcript:
    teacher, conf, note, stats = pick_teacher(segments)
    if teacher_override:
        match = next((sp for sp in stats if sp.lower() == teacher_override.strip().lower()), None)
        if match:
            teacher, conf, note = match, "high", f"'{match}' set as TEACHER by user override."
    assign_roles(segments, teacher)
    return Transcript(segments, source, teacher, conf, note, stats)


# ---- Rendering for the LLM -----------------------------------------------------
def merge_consecutive(segments: list[Segment], max_gap: float = 2.0, max_len: float = 60.0) -> list[Segment]:
    """Merge back-to-back segments from the same speaker to save tokens."""
    out: list[Segment] = []
    for s in segments:
        if (
            out
            and out[-1].speaker == s.speaker
            and s.start - out[-1].end <= max_gap
            and s.end - out[-1].start <= max_len
        ):
            out[-1] = Segment(out[-1].start, s.end, s.speaker, f"{out[-1].text} {s.text}".strip(), s.role)
        else:
            out.append(Segment(s.start, s.end, s.speaker, s.text, s.role))
    return out


def render_for_llm(tr: Transcript, use_names: bool = True) -> str:
    lines = []
    for s in merge_consecutive(tr.segments):
        if s.role == TEACHER:
            who = "TEACHER"
        else:
            name = s.speaker if (use_names and tr.source == "vtt") else ""
            who = f"STUDENT({name})" if name else "STUDENT"
        lines.append(f"[{fmt_ts(s.start)}] {who}: {s.text}")
    return "\n".join(lines)


def class_dynamics(tr: Transcript) -> dict:
    """Simple, explainable talk-time statistics for the report and the LLM."""
    t_secs = sum(s.duration for s in tr.segments if s.role == TEACHER)
    s_secs = sum(s.duration for s in tr.segments if s.role == STUDENT)
    total = (t_secs + s_secs) or 1.0
    student_speakers = {s.speaker for s in tr.segments if s.role == STUDENT}
    teacher_q = sum(s.text.count("?") for s in tr.segments if s.role == TEACHER)
    minutes = max(tr.duration / 60.0, 1e-6)
    return {
        "duration_min": round(tr.duration / 60.0, 1),
        "teacher_talk_pct": round(100 * t_secs / total, 1),
        "student_talk_pct": round(100 * s_secs / total, 1),
        "teacher_turns": sum(1 for s in tr.segments if s.role == TEACHER),
        "student_turns": sum(1 for s in tr.segments if s.role == STUDENT),
        "distinct_student_labels": len(student_speakers),
        "teacher_questions": teacher_q,
        "teacher_questions_per_min": round(teacher_q / minutes, 2),
    }
