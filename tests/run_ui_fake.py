"""Launch the UI with the LLM replaced by a fake (for UI development without an API key)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CACHE_DIR", str(Path("samples/_uicache").resolve()))

from classeval import llm_eval  # noqa: E402
from tests.fake_llm import fake_evaluate  # noqa: E402

llm_eval.evaluate = fake_evaluate

import app  # noqa: E402
from classeval import config, report_html  # noqa: E402

app.demo.queue().launch(
    server_port=int(os.getenv("PORT", "7861")),
    allowed_paths=[str(config.CACHE_DIR), str(config.ASSETS_DIR)],
    css=report_html.CSS,
    theme=app.gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
)
