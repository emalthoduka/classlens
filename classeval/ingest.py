"""Getting the recording onto local disk: uploads and Google Drive / direct links.

Downloads are streamed in chunks straight to disk, so even multi-GB files never
sit in RAM.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import requests

from .errors import UserFacingError

Progress = Callable[[float, str], None]

_DRIVE_ID_PATTERNS = [
    re.compile(r"/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"/d/([A-Za-z0-9_-]{10,})"),
]


def drive_file_id(url: str) -> str | None:
    url = url.strip()
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc and "drive.usercontent.google.com" not in parsed.netloc:
        return None
    for pat in _DRIVE_ID_PATTERNS:
        m = pat.search(parsed.path)
        if m:
            return m.group(1)
    q = parse_qs(parsed.query)
    if q.get("id"):
        return q["id"][0]
    return None


def source_key_for_url(url: str) -> str:
    """Stable cache key for a link, available *before* downloading."""
    fid = drive_file_id(url)
    return f"gdrive:{fid}" if fid else f"url:{url.strip()}"


def _filename_from_headers(resp: requests.Response, fallback: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    name = m.group(1) if m else fallback
    return re.sub(r"[^\w.\- ]+", "_", name)[:120] or fallback


def _stream_to_disk(resp: requests.Response, dest_dir: Path, fallback_name: str, progress: Progress) -> Path:
    name = _filename_from_headers(resp, fallback_name)
    if not Path(name).suffix:
        name += ".mp4"
    dest = dest_dir / name
    total = int(resp.headers.get("content-length") or 0)
    done = 0
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            mb = done / 1024**2
            if total:
                progress(done / total, f"Downloading… {mb:,.0f} / {total / 1024**2:,.0f} MB")
            else:
                progress(0.0, f"Downloading… {mb:,.0f} MB")
    if done == 0:
        raise UserFacingError("The download was empty. Check that the link points to a video file.")
    return dest


def _drive_confirm_form(page: str) -> tuple[str, dict] | None:
    """Large Drive files return an HTML 'can't scan for viruses' page with a form."""
    form = re.search(r'<form[^>]+action="([^"]+)"', page)
    if not form:
        return None
    inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
    return html.unescape(form.group(1)), inputs


def download_from_url(url: str, dest_dir: Path, progress: Progress) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (classeval)"
    fid = drive_file_id(url)
    try:
        if fid:
            resp = session.get(
                "https://drive.usercontent.google.com/download",
                params={"id": fid, "export": "download", "confirm": "t"},
                stream=True,
                timeout=60,
            )
            if "text/html" in resp.headers.get("content-type", ""):
                page = resp.text
                form = _drive_confirm_form(page)
                if form:
                    action, params = form
                    resp = session.get(action, params=params, stream=True, timeout=60)
                if "text/html" in resp.headers.get("content-type", ""):
                    raise UserFacingError(
                        "Google Drive did not return the file. Make sure sharing is set to "
                        "'Anyone with the link can view' and the link points to a single file."
                    )
            resp.raise_for_status()
            return _stream_to_disk(resp, dest_dir, f"{fid}.mp4", progress)

        if not url.lower().startswith(("http://", "https://")):
            raise UserFacingError("Please paste a full http(s) link (Google Drive or a direct video URL).")
        resp = session.get(url, stream=True, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            raise UserFacingError("That link returned a web page, not a video file. Use a direct download link.")
        return _stream_to_disk(resp, dest_dir, "video.mp4", progress)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        hint = " The file may be private or deleted." if code in (401, 403, 404) else ""
        raise UserFacingError(f"Download failed with HTTP {code}.{hint}") from e
    except requests.RequestException as e:
        raise UserFacingError(f"Download failed: {e.__class__.__name__}. Check the link and retry.") from e
