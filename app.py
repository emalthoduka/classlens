"""Gradio UI for the class quality evaluator (Hugging Face Space entry point)."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import gradio as gr

from classeval import config, report_html
from classeval.cache import StageCache, _json_default
from classeval.errors import UserFacingError
from classeval.pipeline import Job, Tracker, run

EMPTY = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def _gallery(rep: dict) -> list[tuple[str, str]]:
    v = rep.get("video") or {}
    checks = (v.get("metrics", {}).get("brand_background") or {}).get("checks", [])
    return [(c["thumb"], f"{c['timestamp']} · {'✓ brand background' if c['found'] else '✗ not found'}")
            for c in checks if Path(c["thumb"]).exists()]


def _outputs(status_html: str, rep: dict | None = None, json_path: str | None = None):
    if rep is None:
        return status_html, EMPTY, EMPTY, EMPTY, EMPTY, [], None, gr.update(visible=False)
    return (
        status_html,
        report_html.hero(rep),
        report_html.teaching(rep),
        report_html.metrics(rep.get("audio"), "Audio quality"),
        report_html.metrics(rep.get("video"), "Video quality"),
        _gallery(rep),
        json_path,
        gr.update(visible=True),
    )


def evaluate(video_file, url):
    url = (url or "").strip()
    if not video_file and not url:
        raise gr.Error("Upload a recording or paste a Google Drive / video link first.")
    job = Job(video_path=None if url else video_file, url=url or None)
    tracker = Tracker()
    box: dict = {}

    def worker():
        try:
            box["report"] = run(job, tracker)
        except UserFacingError as e:
            box["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - surface anything unexpected, without a traceback
            box["error"] = f"Unexpected error ({e.__class__.__name__}): {e}"

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while th.is_alive():
        yield _outputs(report_html.status(tracker.snapshot(), tracker.overall(), time.time() - tracker.t0))
        th.join(timeout=0.5)

    elapsed = time.time() - tracker.t0
    if "error" in box:
        yield _outputs(report_html.status(tracker.snapshot(), tracker.overall(), elapsed, box["error"]))
        raise gr.Error(box["error"], duration=None)

    rep = box["report"]
    sc = StageCache(rep["key"])
    out = sc.path("report.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    yield _outputs(report_html.status(tracker.snapshot(), 1.0, elapsed), rep, str(out))


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


def build() -> gr.Blocks:
    with gr.Blocks(title="ClassLens - Class Quality Evaluator") as demo:
        gr.Markdown("# ClassLens · Class Quality Evaluator\n" + INTRO)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Upload recording"):
                        video = gr.File(label="Class recording (mp4, mkv, mov, webm…)",
                                        file_types=["video", ".mkv", ".webm", ".m4a", ".mp3"], type="filepath")
                    with gr.Tab("Google Drive / link"):
                        url = gr.Textbox(label="Google Drive share link or direct video URL",
                                         placeholder="https://drive.google.com/file/d/…/view?usp=sharing")
                        gr.Markdown("<span style='opacity:.7'>If a link is given it takes priority over an upload.</span>")
                go =gr.Button("Evaluate class", variant="primary", size="lg")
                with gr.Accordion("How scoring works", open=False):
                    gr.Markdown(HOW)
            with gr.Column(scale=4):
                status = gr.HTML(report_html.status(Tracker().snapshot(), 0.0, 0.0))

        with gr.Column(visible=False) as results:
            hero = gr.HTML()
            with gr.Tabs():
                with gr.Tab("Teaching quality (80%)"):
                    teaching = gr.HTML()
                with gr.Tab("Audio quality (10%)"):
                    audio = gr.HTML()
                with gr.Tab("Video quality (10%)"):
                    video_html = gr.HTML()
                    gallery = gr.Gallery(label="Frames sampled while the teacher was speaking (green box = academy logo found)",
                                         columns=4, height="auto", object_fit="contain")
            report_file = gr.File(label="Download full report (JSON)", interactive=False)

        go.click(
            evaluate,
            inputs=[video, url],
            outputs=[status, hero, teaching, audio, video_html, gallery, report_file, results],
            show_progress="hidden",
            concurrency_limit=2,
        )
    return demo


demo = build()

if __name__ == "__main__":
    auth = (config.APP_USERNAME, config.APP_PASSWORD) if config.APP_USERNAME and config.APP_PASSWORD else None
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=2, max_size=20).launch(
        auth=auth,
        auth_message="Enter the team credentials to use the evaluator.",
        max_file_size=config.MAX_UPLOAD,
        allowed_paths=[str(config.CACHE_DIR), str(config.ASSETS_DIR)],
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=report_html.CSS,
        show_error=True,
    )
