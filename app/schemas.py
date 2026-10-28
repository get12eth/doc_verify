from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["match", "review", "no_match", "unavailable"]


class FaceCandidateComparison(BaseModel):
    candidate_a: int
    candidate_b: int
    page_a: int
    page_b: int
    box_a: list[float]
    box_b: list[float]
    similarity: float
    decision: Decision


class FaceDocumentConsistency(BaseModel):
    filename: str
    faces_found: int
    decision: Decision
    comparisons: list[FaceCandidateComparison] = Field(default_factory=list)


class SignaturePairResult(BaseModel):
    signatures: list[int] = Field(description="One-based signature slot numbers compared.")
    pages: list[int] = Field(description="Zero-based page indexes for the compared signatures.")
    score: float
    decision: Decision


class DocumentSignatureResult(BaseModel):
    filename: str
    page: int | None = Field(None, description="Zero-based specimen-signature page.")
    decision: Decision
    comparisons: list[SignaturePairResult] = Field(default_factory=list)


class FaceResult(BaseModel):
    decision: Decision
    similarity: float | None = Field(None, description="Cosine similarity of ArcFace embeddings (-1..1).")
    faces_found_a: int
    faces_found_b: int
    page_a: int | None = Field(None, description="Zero-based page used from document A.")
    page_b: int | None = Field(None, description="Zero-based page used from document B.")
    candidate_comparisons: list[FaceCandidateComparison] = Field(default_factory=list)
    within_document_a: list[FaceCandidateComparison] = Field(default_factory=list)
    within_document_b: list[FaceCandidateComparison] = Field(default_factory=list)
    within_document_decision: Decision = "unavailable"
    within_documents: list[FaceDocumentConsistency] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SignatureResult(BaseModel):
    decision: Decision
    minimum_pair_score: float | None = Field(None, ge=0, le=1, description="Lowest available within-document signature-pair similarity.")
    method: Literal["heuristic", "siamese", "sigverse"] = "heuristic"
    documents: list[DocumentSignatureResult] = Field(default_factory=list)
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


class MultiCompareResult(BaseModel):
    request_id: str
    document_count: int
    overall: Decision
    comparisons: list[PairCompareResult]
    face_consistency: list[FaceDocumentConsistency] = Field(default_factory=list)
    signature_consistency: SignatureResult
    disclaimer: str = "Automated similarity score for decision support only; not forensic or legal proof."
