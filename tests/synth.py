"""Synthetic Zoom-like frames for testing the brand background detector."""
from __future__ import annotations

import cv2
import numpy as np

from classeval import config

RNG = np.random.default_rng(7)


def ref() -> np.ndarray:
    return cv2.imread(str(config.BRAND_BACKGROUND), cv2.IMREAD_COLOR)


def cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale-to-cover then centre-crop, like a video-call virtual background."""
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(np.ceil(img.shape[1] * s)), int(np.ceil(img.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0 : y0 + h, x0 : x0 + w].copy()


def letterbox(img: np.ndarray, w: int, h: int, logo_frac: float = 0.45) -> np.ndarray:
    """Logo placed smaller on a dark 16:9 canvas (another plausible background layout)."""
    canvas = np.full((h, w, 3), 40, np.uint8)
    s = h * logo_frac * 2 / img.shape[0]
    r = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0 : y0 + r.shape[0], x0 : x0 + r.shape[1]] = r
    return canvas


def room(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y] = (90 + 60 * y / h, 110 + 40 * y / h, 130)
    for _ in range(6):
        x, y = RNG.integers(0, w), RNG.integers(0, h)
        cv2.rectangle(img, (int(x), int(y)), (int(x + RNG.integers(30, w // 3)), int(y + RNG.integers(30, h // 3))),
                      tuple(int(c) for c in RNG.integers(30, 220, 3)), -1)
    return img


def person(img: np.ndarray, cx: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    c = (int(w * cx), int(h * 0.55))
    cv2.ellipse(out, (c[0], h), (int(w * 0.28), int(h * 0.30)), 0, 180, 360, (60, 60, 140), -1)  # shoulders
    cv2.ellipse(out, c, (int(h * 0.17), int(h * 0.23)), 0, 0, 360, (140, 170, 210), -1)  # head
    return out


def chessboard(w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 48, np.uint8)
    n = min(w, h) - 40
    sq = n // 8
    x0, y0 = (w - 8 * sq) // 2, (h - 8 * sq) // 2
    for r in range(8):
        for c in range(8):
            col = (181, 217, 240) if (r + c) % 2 == 0 else (99, 136, 181)
            cv2.rectangle(img, (x0 + c * sq, y0 + r * sq), (x0 + (c + 1) * sq, y0 + (r + 1) * sq), col, -1)
            if RNG.random() < 0.3:
                pc = (250, 250, 250) if RNG.random() < 0.5 else (20, 20, 20)
                cv2.circle(img, (x0 + c * sq + sq // 2, y0 + r * sq + sq // 2), sq // 3, pc, -1)
    for i, ch in enumerate("abcdefgh"):
        cv2.putText(img, ch, (x0 + i * sq + sq // 2 - 5, y0 + 8 * sq + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return img


def degrade(img: np.ndarray, q: int = 55, noise: float = 4.0) -> np.ndarray:
    n = img.astype(np.float32) + RNG.normal(0, noise, img.shape)
    ok, buf = cv2.imencode(".jpg", np.clip(n, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def teacher_tile(w: int, h: int, brand: bool, layout: str = "cover", cx: float = 0.5) -> np.ndarray:
    if not brand:
        return person(room(w, h), cx)
    bg = cover(ref(), w, h) if layout == "cover" else letterbox(ref(), w, h)
    return person(bg, cx)


def gallery(W: int, H: int, brand: bool, teacher_idx: int = 4, layout: str = "cover") -> np.ndarray:
    frame = np.full((H, W, 3), 26, np.uint8)
    tw, th = W // 3 - 8, H // 3 - 8
    for i in range(9):
        r, c = divmod(i, 3)
        tile = teacher_tile(tw, th, brand, layout) if i == teacher_idx else person(room(tw, th), RNG.uniform(0.35, 0.65))
        y, x = 4 + r * (th + 8), 4 + c * (tw + 8)
        frame[y : y + th, x : x + tw] = tile
    return frame


def screenshare(W: int, H: int, brand: bool, thumb_w: int = 240, layout: str = "cover") -> np.ndarray:
    frame = np.full((H, W, 3), 26, np.uint8)
    frame[:, : W - thumb_w - 10] = chessboard(W - thumb_w - 10, H)
    th = int(thumb_w * 9 / 16)
    for i in range(min(4, (H - 10) // (th + 6))):
        tile = teacher_tile(thumb_w, th, brand, layout) if i == 0 else person(room(thumb_w, th))
        y = 10 + i * (th + 6)
        frame[y : y + th, W - thumb_w - 5 : W - 5] = tile
    return frame


def scenes() -> list[tuple[str, np.ndarray, bool]]:
    W, H = 1280, 720
    return [
        ("full_cover", degrade(teacher_tile(W, H, True, "cover")), True),
        ("full_letterbox", degrade(teacher_tile(W, H, True, "letterbox")), True),
        ("full_offcentre", degrade(teacher_tile(W, H, True, "cover", cx=0.35)), True),
        ("gallery_cover", degrade(gallery(W, H, True)), True),
        ("gallery_letterbox", degrade(gallery(W, H, True, 2, "letterbox")), True),
        ("share_thumb_320", degrade(screenshare(W, H, True, 320)), True),
        ("share_thumb_240", degrade(screenshare(W, H, True, 240)), True),
        ("share_thumb_240_lb", degrade(screenshare(W, H, True, 240, "letterbox")), True),
        ("neg_full_room", degrade(teacher_tile(W, H, False)), False),
        ("neg_gallery", degrade(gallery(W, H, False)), False),
        ("neg_share", degrade(screenshare(W, H, False)), False),
        ("neg_chessboard", degrade(chessboard(W, H)), False),
        ("neg_black", degrade(np.full((H, W, 3), 40, np.uint8)), False),
    ]
