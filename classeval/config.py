"""Central configuration: models, weights, thresholds and paths.

Everything tunable lives here so scoring can be recalibrated without touching
pipeline code. Secrets are read from the environment only (never hard-coded).
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # local development convenience; on Streamlit Cloud secrets are env vars already
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
BRAND_BACKGROUND = ASSETS_DIR / "brand_background.webp"


def _default_cache_dir() -> Path:
    # HF Spaces persistent storage is mounted at /data when enabled.
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return Path("/data/classeval_cache")
    return ROOT / ".cache"


CACHE_DIR = Path(os.getenv("CACHE_DIR", str(_default_cache_dir())))
WORK_DIR = Path(os.getenv("WORK_DIR", str(CACHE_DIR / "_work")))
CACHE_MAX_GB = float(os.getenv("CACHE_MAX_GB", "5"))

# ---- OpenAI -----------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe-diarize")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium")
# Diarization runs at roughly half real-time, so short chunks processed in parallel are
# much faster than long ones (3-min chunk ~ 1.5 min; a 54-min class ~ 5-6 min total).
TRANSCRIBE_CHUNK_SECONDS = int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "180"))
TRANSCRIBE_PARALLELISM = int(os.getenv("TRANSCRIBE_PARALLELISM", "8"))
TRANSCRIBE_TIMEOUT = float(os.getenv("TRANSCRIBE_TIMEOUT", "300"))
TRANSCRIBE_LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "")  # "" = auto (classes are code-mixed)
PROMPT_VERSION = "v2"  # bump to invalidate cached LLM evaluations
VIDEO_EVAL_VERSION = "v2"  # bump to invalidate cached video scores

# ---- Access control (optional) ---------------------------------------------
APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# ---- Final score composition -----------------------------------------------
PILLAR_WEIGHTS = {"teaching": 0.80, "audio": 0.10, "video": 0.10}

# Lower bound (inclusive) of each class on the 0-100 final score.
CLASS_THRESHOLDS = [("Excellent", 80.0), ("Good", 70.0), ("Average", 50.0), ("Poor", 0.0)]

# A severe teacher incident (shouting, abusive language) caps the class here.
SEVERE_INCIDENT_CAP = "Average"

# ---- Teaching-quality criteria (LLM) ----------------------------------------
# key: (display name, weight, kind). kind:
#   core  - always applicable, absence scores low
#   cond  - only applicable when the situation arises (e.g. a new student joined)
#   bonus - only scored when present; absence is not penalised
TEACHING_CRITERIA: dict[str, tuple[str, float, str]] = {
    "opening_rapport": ("Opening greeting & rapport", 8, "core"),
    "new_student_welcome": ("New student introduction", 4, "cond"),
    "revision_previous_class": ("Revision or warm-up at start", 7, "core"),
    "homework_review": ("Homework discussed", 5, "bonus"),
    "homework_assignment": ("Homework given", 5, "bonus"),
    "closing_recap": ("Closing recap", 7, "core"),
    "tone_warmth": ("Positive tone & warmth", 9, "core"),
    "energy_enthusiasm": ("Energy & enthusiasm", 6, "core"),
    "patience": ("Patience", 7, "core"),
    "praise_encouragement": ("Praise & encouragement", 7, "core"),
    "student_engagement": ("Student engagement", 9, "core"),
    "questioning_frequency": ("Frequent questioning", 6, "core"),
    "individual_questioning": ("Questions to individual students", 6, "core"),
    "concept_clarity": ("Chess concept clarity", 8, "core"),
    "guided_problem_solving": ("Guided problem solving", 7, "core"),
    "classroom_management": ("Classroom management", 5, "core"),
    "humor_laughter": ("Humour & laughter", 3, "bonus"),
    "multimedia_use": ("Topic video shown & discussed", 3, "bonus"),
}

# Points deducted from the 0-100 teaching score per teacher incident.
INCIDENT_PENALTY = {"minor": 2.0, "moderate": 5.0, "severe": 10.0}
INCIDENT_PENALTY_CAP = 25.0

# ---- Audio sub-metric weights (sum need not be 1; normalised) ---------------
AUDIO_WEIGHTS = {
    "loudness": 15,
    "clipping": 15,
    "noise": 25,
    "hum": 5,
    "interruptions": 10,
    "reverb": 15,
    "dropouts": 10,
    "consistency": 5,
}

# ---- Video sub-metric weights ------------------------------------------------
VIDEO_WEIGHTS = {
    "brand_background": 40,
    "resolution": 15,
    "framerate": 10,
    "bitrate": 10,
    "sharpness": 10,
    "noise": 5,
    "compression": 5,
    "exposure": 5,
}
VIDEO_SAMPLE_FRAMES = int(os.getenv("VIDEO_SAMPLE_FRAMES", "16"))
TEACHER_FRAMES = int(os.getenv("TEACHER_FRAMES", "20"))
# Branded background gets full marks once seen in this many teacher-speaking frames
# (1 sighting = 33, 2 = 67, 3+ = 100).
BRAND_FULL_MARKS_HITS = int(os.getenv("BRAND_FULL_MARKS_HITS", "3"))
