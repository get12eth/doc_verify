"""Pluggable learned signature embedders. Each returns an L2-normalised vector, so a dot product is the cosine.

  * SiameseEmbedder  - ONNX model trained with scripts/train_siamese.py (uses our preprocessed input)
  * SigVerseEmbedder - the pretrained `sigverse` package (EfficientNet-B0 Siamese, 256-d, ONNX, MIT licence)
"""
from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from .config import Settings
    from .signature import Prepared


class Embedder(Protocol):
    name: str

    def embed(self, prepared: "Prepared", crop_bgr: np.ndarray) -> np.ndarray: ...


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def session_options(threads: int) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    if threads > 0:                       # cap per-session threads so concurrent requests don't oversubscribe cores
        opts.intra_op_num_threads = threads
    return opts


class SiameseEmbedder:
    name = "siamese"

    def __init__(self, model_path: str, threads: int = 0) -> None:
        self._sess = ort.InferenceSession(model_path, sess_options=session_options(threads),
                                          providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def embed(self, prepared: "Prepared", crop_bgr: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(prepared.net_input[None, None], dtype=np.float32)
        return _unit(self._sess.run(None, {self._input_name: x})[0][0])


class SigVerseEmbedder:
    name = "sigverse"

    def __init__(self) -> None:
        try:
            from sigverse import SignatureEmbedder
        except ImportError as exc:
            raise RuntimeError("DOCVERIFY_SIG_SIGVERSE=true requires `pip install sigverse`.") from exc
        self._emb = SignatureEmbedder()

    def embed(self, prepared: "Prepared", crop_bgr: np.ndarray) -> np.ndarray:
        # SigVerse does its own preprocessing, so it gets the raw crop rather than our cleaned one.
        try:
            return _unit(self._emb.embed(crop_bgr))
        except (TypeError, ValueError, AttributeError):
            # Fallback if this version only accepts file paths: a temp file, deleted immediately.
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                cv2.imwrite(path, crop_bgr)
                return _unit(self._emb.embed(path))
            finally:
                os.unlink(path)


def load_embedder(settings: "Settings") -> Embedder | None:
    if settings.sig_sigverse and settings.sig_siamese_model:
        raise ValueError("Set only one of DOCVERIFY_SIG_SIGVERSE and DOCVERIFY_SIG_SIAMESE_MODEL.")
    if settings.sig_sigverse:
        return SigVerseEmbedder()
    if settings.sig_siamese_model:
        return SiameseEmbedder(settings.sig_siamese_model, settings.ort_threads)
    return None
