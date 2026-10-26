"""Signature localisation with a YOLOv8-style ONNX detector (CPU, ONNX Runtime).

Point DOCVERIFY_SIG_DETECTOR_MODEL at an ONNX export of a single-class signature detector
(search Hugging Face for "signature detection yolov8 onnx" and check the licence). The expected
output layout is the standard YOLOv8 one: (1, 4 + num_classes, num_anchors) with xywh boxes in
letterboxed input pixels. Anything else will fail loudly rather than return wrong boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort

from .embedders import session_options

if TYPE_CHECKING:
    from .config import Settings

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: Box          # relative x0, y0, x1, y1 (0..1)
    score: float


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to size x size. Returns (image, ratio, left, top)."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top: top + nh, left: left + nw] = resized
    return canvas, ratio, left, top


def decode_yolov8(output: np.ndarray, *, conf: float, iou: float, ratio: float,
                  left: int, top: int, width: int, height: int) -> list[Detection]:
    """Turn raw YOLOv8 output into NMS-filtered detections in relative original-image coordinates."""
    preds = np.squeeze(output, axis=0)
    if preds.ndim != 2 or min(preds.shape) < 5:
        raise ValueError(f"Unexpected detector output shape {output.shape}; expected YOLOv8 (1, 4+nc, N).")
    if preds.shape[0] < preds.shape[1]:        # (4+nc, N) -> (N, 4+nc)
        preds = preds.T
    scores = preds[:, 4:].max(axis=1)
    keep = scores >= conf
    if not keep.any():
        return []
    preds, scores = preds[keep], scores[keep]

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x0 = np.clip((cx - bw / 2 - left) / ratio, 0, width)
    y0 = np.clip((cy - bh / 2 - top) / ratio, 0, height)
    x1 = np.clip((cx + bw / 2 - left) / ratio, 0, width)
    y1 = np.clip((cy + bh / 2 - top) / ratio, 0, height)

    xywh = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    out = []
    for i in np.array(idx).reshape(-1):
        if x1[i] - x0[i] < 2 or y1[i] - y0[i] < 2:
            continue
        out.append(Detection((float(x0[i] / width), float(y0[i] / height),
                             float(x1[i] / width), float(y1[i] / height)), float(scores[i])))
    return sorted(out, key=lambda d: d.score, reverse=True)


class SignatureDetector:
    def __init__(self, model_path: str, *, conf: float = 0.35, iou: float = 0.5, pad: float = 0.08,
                 min_y: float = 0.88, threads: int = 0) -> None:
        self._sess = ort.InferenceSession(model_path, sess_options=session_options(threads),
                                          providers=["CPUExecutionProvider"])
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self._size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self._conf, self._iou, self._pad, self._min_y = conf, iou, pad, min_y

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        boxed, ratio, left, top = letterbox(bgr, self._size)
        blob = np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output = self._sess.run(None, {self._input_name: blob})[0]
        detections = decode_yolov8(output, conf=self._conf, iou=self._iou, ratio=ratio,
                       left=left, top=top, width=w, height=h)
        return [d for d in detections if d.box[1] >= self._min_y]

    def best_box(self, bgr: np.ndarray) -> Box | None:
        """Highest-confidence signature, padded so strokes at the edge are not clipped."""
        dets = self.detect(bgr)
        if not dets:
            return None
        x0, y0, x1, y1 = dets[0].box
        px, py = (x1 - x0) * self._pad, (y1 - y0) * self._pad
        return max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)


def load_detector(settings: "Settings") -> SignatureDetector | None:
    if not settings.sig_detector_model:
        return None
    return SignatureDetector(settings.sig_detector_model, conf=settings.sig_detector_conf,
                             pad=settings.sig_detector_pad, min_y=settings.sig_detector_min_y,
                             threads=settings.ort_threads)
