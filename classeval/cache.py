"""Small on-disk cache: one folder per video key, one JSON file per pipeline stage.

Only compact results (JSON + small thumbnails) are kept. Videos and audio are
never cached; they live in a per-job work dir that is deleted after the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from . import config


def partial_file_hash(path: str | Path, block: int = 8 * 1024 * 1024) -> str:
    """Fast content fingerprint: size + first/middle/last 8 MB.

    Hashing a multi-GB recording in full would take tens of seconds; sampling
    three blocks plus the exact size is unique enough for cache keys.
    """
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        for offset in (0, max(0, size // 2 - block // 2), max(0, size - block)):
            f.seek(offset)
            h.update(f.read(block))
    return h.hexdigest()[:24]


def _json_default(o):
    """Make numpy scalars/arrays (which leak out of OpenCV/numpy maths) JSON-serialisable."""
    import numpy as np

    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def text_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


class StageCache:
    def __init__(self, key: str, root: Path | None = None):
        self.key = key
        self.dir = (root or config.CACHE_DIR) / key
        self.dir.mkdir(parents=True, exist_ok=True)
        os.utime(self.dir)  # LRU touch

    def path(self, name: str) -> Path:
        return self.dir / name

    def get(self, stage: str) -> dict | None:
        p = self.dir / f"{stage}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, stage: str, data: dict) -> None:
        p = self.dir / f"{stage}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        tmp.replace(p)


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def evict(max_gb: float | None = None, root: Path | None = None) -> None:
    """Delete least-recently-used cache folders beyond the size budget."""
    root = root or config.CACHE_DIR
    if not root.exists():
        return
    budget = (max_gb if max_gb is not None else config.CACHE_MAX_GB) * 1024**3
    entries = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]
    sized = sorted(((d.stat().st_mtime, _dir_size(d), d) for d in entries), key=lambda t: t[0])
    total = sum(s for _, s, _ in sized)
    for _, size, d in sized:
        if total <= budget:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size


def clean_stale_work_dirs(max_age_hours: float = 6.0) -> None:
    """Remove work dirs left behind by crashed/aborted jobs."""
    if not config.WORK_DIR.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for d in config.WORK_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
