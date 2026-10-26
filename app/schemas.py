from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["match", "review", "no_match", "unavailable"]


class FaceResult(BaseModel):
    decision: Decision
    similarity: float | None = Field(None, description="Cosine similarity of ArcFace embeddings (-1..1).")
    faces_found_a: int
    faces_found_b: int
    warnings: list[str] = Field(default_factory=list)


class SignatureResult(BaseModel):
    decision: Decision
    score: float | None = Field(None, ge=0, le=1, description="Similarity (0..1) from the active method.")
    method: Literal["heuristic", "siamese", "sigverse"] = "heuristic"
    components: dict[str, float] = Field(default_factory=dict)
    box_source: Literal["manual", "auto", "none"] = "none"
    box_a: list[float] | None = Field(None, description="Relative x0,y0,x1,y1 used for document A.")
    box_b: list[float] | None = None
    warnings: list[str] = Field(default_factory=list)


class CompareResult(BaseModel):
    request_id: str
    overall: Decision
    face: FaceResult
    signature: SignatureResult | None = Field(
        None, description="Unavailable when a signature could not be compared."
    )
    disclaimer: str = "Automated similarity score for decision support only; not forensic or legal proof."


class PairCompareResult(BaseModel):
    document_a: int = Field(description="Zero-based index of the first document.")
    document_b: int = Field(description="Zero-based index of the second document.")
    overall: Decision
    face: FaceResult
    signature: SignatureResult


class MultiCompareResult(BaseModel):
    request_id: str
    document_count: int
    overall: Decision
    comparisons: list[PairCompareResult]
    disclaimer: str = "Automated similarity score for decision support only; not forensic or legal proof."
