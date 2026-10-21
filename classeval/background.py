"""Detect the academy's branded virtual background inside a video frame.

The teacher's camera may be full-frame, a gallery tile, or a small thumbnail
during screen share, and the teacher's body hides part of the background. So
instead of whole-frame comparison we look for the *logo* anywhere in the frame:

1. SIFT keypoints of the logo (scale-invariant, so tile size doesn't matter).
2. Ratio-test matching against the frame, then a RANSAC similarity transform
   (scale + small rotation + shift). Requiring many geometrically consistent
   inliers rejects look-alikes such as a chessboard screen-share, which shares
   the logo's checker texture but not its letter/digit layout.
3. Occlusion is fine: only a subset of logo keypoints needs to be visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config

MIN_INLIERS = 8  # geometric evidence needed before photometric verification
STRONG_INLIERS = 30  # enough on its own even if heavy occlusion lowers correlation
MIN_NCC = 0.35  # correlation of the warped logo with the frame (negatives score ~0)


@dataclass
class Match:
    found: bool
    inliers: int
    scale: float = 0.0
    box: tuple[int, int, int, int] | None = None  # x, y, w, h of the logo in the frame
    mirrored: bool = False
    ncc: float = 0.0


@lru_cache(maxsize=4)
def _reference(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 100)
    pad = 6
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, gray.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, gray.shape[0])
    logo = gray[y0:y1, x0:x1]
    # Upscale the small reference so SIFT finds stable keypoints on letters/digits.
    logo = cv2.resize(logo, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    sift = cv2.SIFT_create()
    refs = []
    for mirrored, im in ((False, logo), (True, cv2.flip(logo, 1))):
        kp, des = sift.detectAndCompute(im, None)
        refs.append((mirrored, kp, des, im))
    return refs


def _frame_features(gray: np.ndarray):
    # Small frames/thumbnails: upscale so a ~60 px logo still yields keypoints.
    scale = 1.0
    if gray.shape[1] < 1600:
        scale = 1600 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    sift = cv2.SIFT_create(nfeatures=6000)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des, scale, gray


def _ncc(template: np.ndarray, M: np.ndarray, gray: np.ndarray) -> float:
    """Correlation between the logo warped into the frame and the frame pixels."""
    warped = cv2.warpAffine(template.astype(np.float32), M, (gray.shape[1], gray.shape[0]),
                            flags=cv2.INTER_LINEAR, borderValue=-1)
    valid = warped >= 0
    if valid.sum() < 100:
        return 0.0
    t, f = warped[valid], gray[valid].astype(np.float32)
    if t.std() < 1e-3 or f.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(t, f)[0, 1])


def detect(frame_bgr: np.ndarray, reference: Path | None = None) -> Match:
    reference = reference or config.BRAND_BACKGROUND
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp_f, des_f, up, gray_up = _frame_features(gray)
    if des_f is None or len(kp_f) < MIN_INLIERS:
        return Match(False, 0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = Match(False, 0)
    for mirrored, kp_r, des_r, tmpl in _reference(str(reference)):
        rh, rw = tmpl.shape[:2]
        if des_r is None:
            continue
        pairs = matcher.knnMatch(des_r, des_f, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            best = max(best, Match(False, len(good)), key=lambda m: m.inliers)
            continue
        src = np.float32([kp_r[m.queryIdx].pt for m in good])
        dst = np.float32([kp_f[m.trainIdx].pt for m in good])
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=4000)
        if M is None:
            continue
        inl = int(mask.sum())
        s = float(np.hypot(M[0, 0], M[1, 0]))
        angle = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        w_px = rw * s / up
        plausible = bool(angle < 12 and 25 <= w_px <= frame_bgr.shape[1] * 1.6)
        ncc = _ncc(tmpl, M, gray_up) if plausible and inl >= MIN_INLIERS else 0.0
        found = bool(plausible and inl >= MIN_INLIERS and (ncc >= MIN_NCC or inl >= STRONG_INLIERS))
        box = None
        if plausible:
            corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
            pts = cv2.transform(corners, M).reshape(-1, 2) / up
            x, y = pts.min(axis=0)
            X, Y = pts.max(axis=0)
            box = (int(x), int(y), int(X - x), int(Y - y))
        cand = Match(found, inl, s / up, box, mirrored, round(ncc, 3))
        if (cand.found, cand.ncc, cand.inliers) > (best.found, best.ncc, best.inliers):
            best = cand
    return best


def annotate(frame_bgr: np.ndarray, m: Match, max_w: int = 480) -> np.ndarray:
    out = frame_bgr.copy()
    if m.box:
        x, y, w, h = m.box
        color = (60, 200, 60) if m.found else (40, 40, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, out.shape[1] // 300))
    if out.shape[1] > max_w:
        out = cv2.resize(out, (max_w, int(out.shape[0] * max_w / out.shape[1])), interpolation=cv2.INTER_AREA)
    return out
