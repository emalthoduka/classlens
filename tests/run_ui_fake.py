"""Launch the UI with the LLM replaced by a fake (for UI development without an API key).

Run with: streamlit run tests/run_ui_fake.py
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CACHE_DIR", str(Path("samples/_uicache").resolve()))

from classeval import llm_eval  # noqa: E402
from tests.fake_llm import fake_evaluate  # noqa: E402

llm_eval.evaluate = fake_evaluate

runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__main__")
