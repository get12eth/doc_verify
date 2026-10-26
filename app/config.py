from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent 


class Settings(BaseSettings):
    """Runtime configuration. Override with env vars prefixed DOCVERIFY_ or a .env file."""

    model_config = SettingsConfigDict(env_prefix="DOCVERIFY_", env_file=".env", extra="ignore")
    
        # --- face ---
    face_model_root: Path = PROJECT_ROOT / "models" / "insightface" 

    # --- upload / decoding limits -------------------------------------------------
    max_upload_mb: int = 15
    max_pixels: int = 60_000_000      # guards against decompression bombs
    pdf_dpi: int = 300
    max_compare_documents: int = 10   # bounds pairwise work to at most 45 comparisons

    # --- concurrency (inference is CPU-bound) -------------------------------------
    max_concurrency: int = 2             # size of the dedicated inference thread pool
    ort_threads: int = 0                 # intra-op threads per ONNX session; 0 = ORT default.
                                         # Tip: cores // max_concurrency avoids CPU oversubscription.

    # --- face ----------------------------------------------------------------------
    face_model_pack: str = "buffalo_l"   # ArcFace R50 (w600k) + SCRFD detector
    face_det_size: int = 1024            # multiple of 32; larger finds small ID-card portraits
    face_det_min_score: float = 0.5
    face_match_threshold: float = 0.45   # cosine similarity; calibrate on your own documents
    face_review_threshold: float = 0.30

    # --- signature (heuristic score in 0..1; calibrate with scripts/calibrate_signature.py)
    sig_match_threshold: float = 0.60
    sig_review_threshold: float = 0.40

    # --- optional signature auto-detector (YOLOv8 ONNX); unset = boxes must be supplied ---------
    sig_detector_model: str | None = None
    sig_detector_conf: float = 0.35
    sig_detector_pad: float = 0.08       # fraction of box size added on each side
    sig_detector_min_y: float = 0.88    # normalized top edge; tune to the document's signature area

    # --- optional trained Siamese embedder (ONNX from scripts/train_siamese.py) -----------------
    # When set, its cosine similarity becomes the signature score and uses the thresholds below
    # (different scale from the heuristic ones above: recalibrate!).
    sig_siamese_model: str | None = None
    sig_siamese_match_threshold: float = 0.80
    sig_siamese_review_threshold: float = 0.65

    # --- optional pretrained verifier: `pip install sigverse`, then DOCVERIFY_SIG_SIGVERSE=true ----
    # Score is cosine similarity of its 256-d embeddings. 0.50 equals SigVerse's own default
    # L2-distance threshold of 1.0 (for unit vectors, distance^2 = 2 - 2*cos). Placeholders: calibrate.
    sig_sigverse: bool = False
    sig_sigverse_match_threshold: float = 0.50
    sig_sigverse_review_threshold: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()
