"""Face verification with InsightFace (SCRFD detector + ArcFace embeddings, CPU/ONNX Runtime)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis

from .config import Settings
from .decision import decide
from .schemas import FaceResult

import warnings

warnings.filterwarnings("ignore", message=r".*`estimate` is deprecated.*", category=FutureWarning)

MIN_FACE_WIDTH_PX = 80


@dataclass(frozen=True)
class FaceCandidate:
    embedding: np.ndarray
    box: tuple[float, float, float, float]


class FaceMatcher:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._app = FaceAnalysis(
            name=settings.face_model_pack,
            root=str(settings.face_model_root),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=-1,
            det_thresh=settings.face_det_min_score,
            det_size=(settings.face_det_size, settings.face_det_size),
        )

    def _largest_face(self, img: np.ndarray):
        faces = self._app.get(img)
        if not faces:
            return None, 0
        # ID cards often carry a small "ghost" portrait: the main photo is the largest face.
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best, len(faces)

    def candidates(self, img: np.ndarray) -> list[FaceCandidate]:
        height, width = img.shape[:2]
        candidates = []
        for face in self._app.get(img):
            x0, y0, x1, y1 = face.bbox
            candidates.append(FaceCandidate(
                embedding=face.normed_embedding,
                box=(float(np.clip(x0 / width, 0, 1)), float(np.clip(y0 / height, 0, 1)),
                     float(np.clip(x1 / width, 0, 1)), float(np.clip(y1 / height, 0, 1))),
            ))
        return candidates

    def decide_similarity(self, similarity: float) -> str:
        return decide(similarity, self._s.face_match_threshold, self._s.face_review_threshold)

    def compare(self, img_a: np.ndarray, img_b: np.ndarray) -> FaceResult:
        (face_a, n_a), (face_b, n_b) = self._largest_face(img_a), self._largest_face(img_b)

        warnings: list[str] = []
        for label, face, n in (("A", face_a, n_a), ("B", face_b, n_b)):
            if face is None:
                warnings.append(f"No face detected in document {label}.")
                continue
            if n > 1:
                warnings.append(f"{n} faces detected in document {label}; the largest was used.")
            if face.bbox[2] - face.bbox[0] < MIN_FACE_WIDTH_PX:
                warnings.append(f"Face in document {label} is small (<{MIN_FACE_WIDTH_PX}px); accuracy is reduced.")

        if face_a is None or face_b is None:
            return FaceResult(decision="unavailable", similarity=None,
                              faces_found_a=n_a, faces_found_b=n_b, warnings=warnings)

        # normed_embedding is L2-normalised, so the dot product is the cosine similarity.
        sim = float(np.clip(np.dot(face_a.normed_embedding, face_b.normed_embedding), -1.0, 1.0))
        return FaceResult(
            decision=decide(sim, self._s.face_match_threshold, self._s.face_review_threshold),
            similarity=round(sim, 4),
            faces_found_a=n_a,
            faces_found_b=n_b,
            warnings=warnings,
        )
