
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from skimage.feature import hog
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize

from .decision import decide
from .schemas import SignatureResult

if TYPE_CHECKING:
    from .config import Settings
    from .embedders import Embedder

CANVAS_W, CANVAS_H, MARGIN = 384, 192, 12
NET_W, NET_H = 192, 96     # input size of the Siamese network (training and serving must match)
MAX_SIDE = 1600           # cap work size for very large crops
MIN_INK_PX = 150          # below this the crop is treated as "no signature found"
LOW_INK_PX = 400
CHAMFER_CLIP_PX = 30.0    # clip outlier distances (stray marks)
CHAMFER_TAU = 8.0         # distance -> similarity: exp(-d / tau)
MAX_SHIFT, SHIFT_STEP = 10, 2
WEIGHTS = {"chamfer": 0.5, "hog": 0.3, "ssim": 0.2}


@dataclass(frozen=True)
class Prepared:
    skeleton: np.ndarray     # bool  (H, W)
    blurred: np.ndarray      # float32 (H, W) thickened + Gaussian-blurred strokes
    dist: np.ndarray         # float32 (H, W) distance to nearest skeleton pixel
    net_input: np.ndarray    # float32 (NET_H, NET_W) in [0, 1]: Siamese network input
    ink_px: int


# --------------------------------------------------------------------------- preprocessing
def _extract_ink(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Divide by an estimate of the paper background to remove shadows / uneven lighting.
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    flat = cv2.GaussianBlur(cv2.divide(gray, background, scale=255), (3, 3), 0)
    if flat.std() < 8:                       # essentially blank: Otsu would just binarise noise
        return None
    _, ink = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #Remove long horizontal ruled lines (signature baselines, form fields).
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 3), 1))
    lines = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, line_kernel), np.ones((3, 1), np.uint8))
    ink = cv2.subtract(ink, lines)

    #Remove specks (vectorised connected-component filter).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return None
    keep = stats[:, cv2.CC_STAT_AREA] >= max(10, int(2e-5 * h * w))
    keep[0] = False                          # background label
    return (keep[labels] * 255).astype(np.uint8)


def _normalise(ink: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(ink)
    if xs.size < MIN_INK_PX:
        return None
    crop = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    ch, cw = crop.shape
    scale = min((CANVAS_W - 2 * MARGIN) / cw, (CANVAS_H - 2 * MARGIN) / ch)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) > 40   # keeps thin strokes
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    x0, y0 = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    canvas[y0: y0 + nh, x0: x0 + nw] = resized
    return canvas


def prepare(bgr: np.ndarray) -> Prepared | None:
    ink = _extract_ink(bgr)
    canvas = _normalise(ink) if ink is not None else None
    if canvas is None:
        return None
    skeleton = skeletonize(canvas)
    thick = cv2.dilate(skeleton.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blurred = cv2.GaussianBlur(thick.astype(np.float32), (0, 0), 3)
    dist = cv2.distanceTransform((~skeleton).astype(np.uint8), cv2.DIST_L2, 3)
    net_input = cv2.resize(thick.astype(np.float32), (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    return Prepared(skeleton=skeleton, blurred=blurred, dist=dist, net_input=net_input,
                    ink_px=int(canvas.sum()))

# --------------------------------------------------------------------------- metrics
def _chamfer_similarity(a: Prepared, b: Prepared) -> float:
    """Symmetric mean nearest-stroke distance, minimised over small translations."""
    pa, pb = np.argwhere(a.skeleton), np.argwhere(b.skeleton)
    h, w = a.skeleton.shape
    best = np.inf
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1, SHIFT_STEP):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1, SHIFT_STEP):
            d_ab = b.dist[np.clip(pa[:, 0] + dy, 0, h - 1), np.clip(pa[:, 1] + dx, 0, w - 1)]
            d_ba = a.dist[np.clip(pb[:, 0] - dy, 0, h - 1), np.clip(pb[:, 1] - dx, 0, w - 1)]
            d = 0.5 * (np.minimum(d_ab, CHAMFER_CLIP_PX).mean() + np.minimum(d_ba, CHAMFER_CLIP_PX).mean())
            best = min(best, d)
    return float(np.exp(-best / CHAMFER_TAU))


def _hog_similarity(a: Prepared, b: Prepared) -> float:
    fa, fb = (hog(p.blurred, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for p in (a, b))
    cos = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-9))
    return float(np.clip((cos - 0.5) / 0.5, 0.0, 1.0))   # HOG is non-negative -> raw cosine has a high floor


def _ssim_similarity(a: Prepared, b: Prepared) -> float:
    _, ssim_map = structural_similarity(a.blurred, b.blurred, data_range=1.0, full=True)
    mask = (a.blurred > 0.02) | (b.blurred > 0.02)        # ignore the empty background
    return float(np.clip(ssim_map[mask].mean(), 0.0, 1.0)) if mask.any() else 0.0


# --------------------------------------------------------------------------- public API
def compare_signatures(img_a: np.ndarray, img_b: np.ndarray, settings: "Settings",
                       embedder: "Embedder | None" = None) -> SignatureResult:
    pa, pb = prepare(img_a), prepare(img_b)
    warnings: list[str] = []
    for label, p in (("A", pa), ("B", pb)):
        if p is None:
            warnings.append(f"No usable signature ink found in {label}.")
        elif p.ink_px < LOW_INK_PX:
            warnings.append(f"Very little ink in {label}; result is unreliable.")
    if pa is None or pb is None:
        return SignatureResult(decision="unavailable", warnings=warnings)

    components = {
        "chamfer": _chamfer_similarity(pa, pb),
        "hog": _hog_similarity(pa, pb),
        "ssim": _ssim_similarity(pa, pb),
    }
    if embedder is not None:   # learned model takes over; heuristics stay as diagnostics
        embedding_a, embedding_b = embedder.embed(pa, img_a), embedder.embed(pb, img_b)
        method = embedder.name
        if method == "siamese":
            distance = float(np.linalg.norm(embedding_a - embedding_b))
            components["euclidean_distance"] = distance
            components[method] = float(np.clip(1.0 - distance / 2.0, 0.0, 1.0))
        else:
            components[method] = float(np.clip(np.dot(embedding_a, embedding_b), 0.0, 1.0))
        score = components[method]
    else:
        method = "heuristic"
        score = float(sum(WEIGHTS[k] * v for k, v in components.items()))
    thresholds = {   # each method lives on its own scale, so each has its own thresholds
        "heuristic": (settings.sig_match_threshold, settings.sig_review_threshold),
        "siamese": (settings.sig_siamese_match_threshold, settings.sig_siamese_review_threshold),
        "sigverse": (settings.sig_sigverse_match_threshold, settings.sig_sigverse_review_threshold),
    }
    match_at, review_at = thresholds[method]

    return SignatureResult(
    
        decision=decide(score, match_at, review_at),
        minimum_pair_score=round(score, 4),
        method=method,
        components={k: round(v, 4) for k, v in components.items()},
        warnings=warnings,
    )
