"""Streamlit UI for the class quality evaluator (Streamlit Community Cloud entry point)."""
from __future__ import annotations

import hmac
import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

from classeval import config, report_html
from classeval.cache import StageCache, _json_default
from classeval.errors import UserFacingError
from classeval.pipeline import Job, Tracker, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

TITLE = "ClassLens - Class Quality Evaluator"
VIDEO_TYPES = ["mp4", "mkv", "mov", "webm", "avi", "m4v", "mpeg", "mpg", "wmv", "flv", "3gp", "ts", "m4a", "mp3"]

# report_html.CSS was written against Gradio's theme variables; define them here with the
# dark values of the Soft (blue / slate) theme the Gradio app used, so the cards look the same.
THEME_VARS = """
:root{--font:inherit;--font-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--body-text-color:#e2e8f0;--body-text-color-subdued:#94a3b8;--block-background-fill:#1e293b;
--border-color-primary:#334155;--background-fill-secondary:#0f172a;}
.ce-note{color:#fbbf24;}
.ce-T{color:#60a5fa;}
.ce-S{color:#c084fc;}
[data-testid="stMain"],[data-testid="stAppViewContainer"]{scroll-behavior:smooth;}
a.ce-scroll{display:flex;align-items:center;gap:14px;margin-top:4px;padding:14px 18px;border-radius:14px;
text-decoration:none;color:var(--body-text-color);border:1px solid rgba(59,130,246,.55);
background:linear-gradient(135deg,rgba(37,99,235,.22),rgba(37,99,235,.06));transition:transform .15s,border-color .15s;}
a.ce-scroll:hover{transform:translateY(-1px);border-color:#60a5fa;color:var(--body-text-color);text-decoration:none;}
.ce-scroll .arrow{flex:none;width:38px;height:38px;border-radius:50%;background:#2563eb;color:#fff;font-size:20px;
display:flex;align-items:center;justify-content:center;animation:cebounce 1.6s ease-in-out infinite;}
.ce-scroll .t1{font-weight:700;}
.ce-scroll .t2{font-size:.85em;color:var(--body-text-color-subdued);margin-top:2px;}
.ce-scroll .lbl{display:inline-block;padding:1px 9px;border-radius:99px;color:#fff;font-size:.85em;margin:0 4px;}
@keyframes cebounce{0%,100%{transform:translateY(0)}50%{transform:translateY(5px)}}
@media (prefers-reduced-motion:reduce){.ce-scroll .arrow{animation:none}}
"""

INTRO = """
Upload a recorded **live class** (or paste a Google Drive link) to get a **Poor / Average / Good / Excellent**
rating with an explanation for every score. Teaching quality is judged from the transcript (80%), audio and
video quality by signal analysis (10% each). Long recordings (45 min+) are supported; results are cached, so re-running
the same video is instant.

"""

HOW = f"""
**Scoring.** Final = 0.8 × Teaching + 0.1 × Audio + 0.1 × Video (each 0–100).
Classes: Excellent ≥ {config.CLASS_THRESHOLDS[0][1]:.0f}, Good ≥ {config.CLASS_THRESHOLDS[1][1]:.0f},
Average ≥ {config.CLASS_THRESHOLDS[2][1]:.0f}, otherwise Poor. A severe teacher incident (e.g. shouting) caps the result at Average.

**Tip.** For files over ~1 GB prefer a Drive link (set to *Anyone with the link*).
"""


@st.cache_resource
def _job_slots() -> threading.BoundedSemaphore:
    # Shared by every session: at most two evaluations run at once, the rest wait their turn.
    return threading.BoundedSemaphore(2)


def _gallery(rep: dict) -> list[tuple[str, str]]:
    v = rep.get("video") or {}
    checks = (v.get("metrics", {}).get("brand_background") or {}).get("checks", [])
    return [(c["thumb"], f"{c['timestamp']} · {'✓ brand background' if c['found'] else '✗ not found'}")
            for c in checks if Path(c["thumb"]).exists()]


def _login() -> None:
    """Shared username/password gate, enabled when APP_USERNAME and APP_PASSWORD are both set."""
    if not (config.APP_USERNAME and config.APP_PASSWORD) or st.session_state.get("authed"):
        return
    st.markdown("# ClassLens · Class Quality Evaluator")
    with st.form("login"):
        st.caption("Enter the team credentials to use the evaluator.")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login", type="primary"):
            ok_user = hmac.compare_digest(user.encode(), config.APP_USERNAME.encode())
            ok_pwd = hmac.compare_digest(pwd.encode(), config.APP_PASSWORD.encode())
            if ok_user and ok_pwd:
                st.session_state.authed = True
                st.rerun()
            st.error("Incorrect username or password.")
    st.stop()


def _save_upload(upload) -> tuple[str, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="classeval_upload_"))
    path = tmp / Path(upload.name).name
    with path.open("wb") as out:
        shutil.copyfileobj(upload, out, 1 << 20)
    return str(path), tmp


def evaluate(upload, url: str, status_box) -> None:
    for k in ("report", "json_path", "error"):
        st.session_state.pop(k, None)
    tmp_dir = None
    if url:
        job = Job(url=url)
    else:
        video_path, tmp_dir = _save_upload(upload)
        job = Job(video_path=video_path)
    tracker = Tracker()
    slots = _job_slots()
    box: dict = {}

    def worker():
        try:
            with slots:
                box["report"] = run(job, tracker)
        except UserFacingError as e:
            box["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - surface anything unexpected, without a traceback
            box["error"] = f"Unexpected error ({e.__class__.__name__}): {e}"
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while th.is_alive():
        status_box.html(report_html.status(tracker.snapshot(), tracker.overall(), time.time() - tracker.t0))
        th.join(timeout=0.5)

    elapsed = time.time() - tracker.t0
    if "error" in box:
        st.session_state.status_html = report_html.status(tracker.snapshot(), tracker.overall(), elapsed, box["error"])
        st.session_state.error = box["error"]
        return

    rep = box["report"]
    out = StageCache(rep["key"]).path("report.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    st.session_state.status_html = report_html.status(tracker.snapshot(), 1.0, elapsed)
    st.session_state.report = rep
    st.session_state.json_path = str(out)


def _scroll_hint(rep: dict) -> str:
    f = rep["final"]
    color = report_html.CLASS_COLORS[f["label"]]
    return (f'<a class="ce-scroll" href="#ce-results"><span class="arrow">↓</span><div>'
            f'<div class="t1">Your report is ready:<span class="lbl" style="background:{color}">{f["label"]}</span>'
            f'{f["score"]:.1f}/100</div>'
            '<div class="t2">Scroll down for the full breakdown: teaching, audio and video scores with the reasons '
            'behind each.</div></div></a>')


def results(rep: dict) -> None:
    st.html('<div id="ce-results"></div>')
    st.html(report_html.hero(rep))
    t_teach, t_audio, t_video = st.tabs(["Teaching quality ", "Audio quality ", "Video quality & Logo detection "])
    with t_teach:
        st.html(report_html.teaching(rep))
    with t_audio:
        st.html(report_html.metrics(rep.get("audio"), "Audio quality"))
    with t_video:
        st.html(report_html.metrics(rep.get("video"), "Video quality"))
        frames = _gallery(rep)
        if frames:
            st.caption("Frames sampled while the teacher was speaking (green box = academy logo found)")
            cols = st.columns(4)
            for i, (thumb, caption) in enumerate(frames):
                cols[i % 4].image(thumb, caption=caption, width="stretch")
    json_path = Path(st.session_state.get("json_path") or "")
    if json_path.is_file():
        st.download_button("Download full report (JSON)", json_path.read_bytes(), file_name="report.json",
                           mime="application/json", on_click="ignore")


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.html(f"<style>{THEME_VARS}{report_html.CSS}</style>")
    _login()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    st.markdown("# ClassLens · Class Quality Evaluator\n" + INTRO)

    left, right = st.columns([5, 4])
    with left:
        tab_upload, tab_link = st.tabs(["Upload recording", "Google Drive / link"])
        with tab_upload:
            upload = st.file_uploader("Class recording (mp4, mkv, mov, webm…)", type=VIDEO_TYPES)
        with tab_link:
            url = st.text_input("Google Drive share link or direct video URL",
                                placeholder="https://drive.google.com/file/d/…/view?usp=sharing")
            st.caption("If a link is given it takes priority over an upload.")
        go = st.button("Evaluate class", type="primary", width="stretch")
        message = st.empty()
        with st.expander("How scoring works"):
            st.markdown(HOW)
        hint_box = st.empty()
    with right:
        status_box = st.empty()
        status_box.html(st.session_state.get("status_html") or report_html.status(Tracker().snapshot(), 0.0, 0.0))

    if go:
        url = (url or "").strip()
        if not upload and not url:
            st.session_state.error = "Upload a recording or paste a Google Drive / video link first."
        else:
            evaluate(upload, url, status_box)
            status_box.html(st.session_state.status_html)

    if st.session_state.get("error"):
        message.error(st.session_state.error)
    if st.session_state.get("report"):
        hint_box.html(_scroll_hint(st.session_state.report))
        results(st.session_state.report)


main()
