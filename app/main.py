from __future__ import annotations

import asyncio
import functools
import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from itertools import combinations
from typing import Annotated, TypeVar

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from .config import Settings, get_settings
from .decision import combine, decide
from .detector import load_detector
from .embedders import load_embedder
from .face import FaceMatcher
from .imaging import InvalidDocument, load_document_pages
from .schemas import (CompareResult, DocumentSignatureResult, FaceCandidateComparison,
                      FaceDocumentConsistency, FaceResult, MultiCompareResult, PairCompareResult,
                      SignaturePairResult, SignatureResult)
from .specimen import find_three_signature_row
from .signature import compare_signatures, prepare

log = logging.getLogger("docverify")
T = TypeVar("T")
Box = tuple[float, float, float, float]
MAX_CANDIDATES_PER_DOCUMENT = 20

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Load models once, off the event loop (first run downloads the InsightFace pack).
    app.state.faces = await run_in_threadpool(FaceMatcher, settings)
    app.state.detector = await run_in_threadpool(load_detector, settings)     # None if not configured
    app.state.embedder = await run_in_threadpool(load_embedder, settings)     # None if not configured
    # Dedicated, bounded pool for all CPU-bound work (decoding, detection, embedding). Isolated from the
    # default pool, and its size is the concurrency limit: extra requests simply queue.
    app.state.pool = ThreadPoolExecutor(max_workers=settings.max_concurrency, thread_name_prefix="docverify")
    try:
        yield
    finally:
        app.state.pool.shutdown(wait=False, cancel_futures=True)

app = FastAPI(
    title="DocVerify",
    version="1.0.0",
    description="Compare the portrait photo and handwritten signature of two documents. "
                "Scores are decision support, not proof.",
    lifespan=lifespan,
)
SettingsDep = Annotated[Settings, Depends(get_settings)]


# --------------------------------------------------------------------------- helpers
async def _read(upload: UploadFile, s: Settings) -> bytes:
    # Note: enforce the hard size limit at your reverse proxy too (e.g. nginx client_max_body_size).
    limit = s.max_upload_mb * 1024 * 1024
    data = await upload.read(limit + 1)
    if not data:
        raise HTTPException(400, f"'{upload.filename}' is empty.")
    if len(data) > limit:
        raise HTTPException(413, f"'{upload.filename}' exceeds {s.max_upload_mb} MB.")
    return data


async def _load_pages(request: Request, upload: UploadFile, page: int, s: Settings) -> list[np.ndarray]:
    data = await _read(upload, s)
    try:
        return await _cpu(request, functools.partial(load_document_pages, data, page=page, dpi=s.pdf_dpi,
                                                     max_pixels=s.max_pixels))
    except InvalidDocument as exc:
        raise HTTPException(422, f"{upload.filename}: {exc}") from exc


async def _cpu(request: Request, fn: Callable[..., T], *args) -> T:
    """Run CPU-bound work on the dedicated inference pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(request.app.state.pool, functools.partial(fn, *args))


def _parse_box(raw: str | None, name: str) -> Box | None:
    """'x0,y0,x1,y1' as fractions (0..1) of the page width/height."""
    if not raw:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in raw.split(","))
    except ValueError:
        raise HTTPException(422, f"{name} must be 'x0,y0,x1,y1' (four numbers).") from None
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise HTTPException(422, f"{name} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1.")
    return x0, y0, x1, y1


def _crop(img: np.ndarray, box: Box | None, name: str) -> np.ndarray:
    if box is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    crop = img[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)]
    if crop.size == 0:
        raise HTTPException(422, f"{name} selects an empty region.")
    return crop


async def _signature_check_pages(request: Request, s: Settings, pages_a: list[np.ndarray],
                                 pages_b: list[np.ndarray] | None,
                                 filename_a: str | None = None,
                                 filename_b: str | None = None) -> SignatureResult:
    async def collect_pages(pages):
        if pages is None:
            return []
        candidates = []
        for page_number, image in enumerate(pages):
            row = await _cpu(request, find_three_signature_row, image)
            if row is not None:
                candidates.extend((1.0, page_number, image, box) for box in row)
        return candidates

    candidates_a = await collect_pages(pages_a)
    candidates_b = await collect_pages(pages_b)

    candidates_a.sort(key=lambda candidate: candidate[0], reverse=True)
    candidates_b.sort(key=lambda candidate: candidate[0], reverse=True)
    truncated = (len(candidates_a) > MAX_CANDIDATES_PER_DOCUMENT
                 or len(candidates_b) > MAX_CANDIDATES_PER_DOCUMENT)
    candidates_a = candidates_a[:MAX_CANDIDATES_PER_DOCUMENT]
    candidates_b = candidates_b[:MAX_CANDIDATES_PER_DOCUMENT]

    def reading_order(candidate):
        _, page_number, _, (x0, y0, x1, y1) = candidate
        return page_number, (y0 + y1) / 2, (x0 + x1) / 2

    candidates_a.sort(key=reading_order)
    candidates_b.sort(key=reading_order)

    warnings = []
    if not candidates_a:
        warnings.append(f"No clear row of three signature-like ink groups found in {filename_a or 'document A'}.")
    if pages_b is not None and not candidates_b:
        warnings.append(f"No clear row of three signature-like ink groups found in {filename_b or 'document B'}.")
    if truncated:
        warnings.append(f"Candidate comparisons are limited to the top {MAX_CANDIDATES_PER_DOCUMENT} detections per document.")
    def padded(box: Box) -> Box:
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * s.sig_detector_pad, (y1 - y0) * s.sig_detector_pad
        return max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)

    prepared_a = []
    prepared_b = []
    for candidate_a, (confidence_a, page_a, image_a, raw_box_a) in enumerate(candidates_a):
        box_a = padded(raw_box_a)
        crop_a = _crop(image_a, box_a, "box_a")
        prepared_a.append((candidate_a, confidence_a, page_a, box_a, crop_a,
                           await _cpu(request, prepare, crop_a)))
    for candidate_b, (confidence_b, page_b, image_b, raw_box_b) in enumerate(candidates_b):
        box_b = padded(raw_box_b)
        crop_b = _crop(image_b, box_b, "box_b")
        prepared_b.append((candidate_b, confidence_b, page_b, box_b, crop_b,
                           await _cpu(request, prepare, crop_b)))

    embedder = request.app.state.embedder
    embeddings_a = [embedder.embed(item[5], item[4]) if item[5] is not None else None
                    for item in prepared_a] if embedder is not None else None
    embeddings_b = [embedder.embed(item[5], item[4]) if item[5] is not None else None
                    for item in prepared_b] if embedder is not None else None
    if embedder is not None:
        thresholds = {
            "siamese": (s.sig_siamese_match_threshold, s.sig_siamese_review_threshold),
            "sigverse": (s.sig_sigverse_match_threshold, s.sig_sigverse_review_threshold),
        }
        match_at, review_at = thresholds[embedder.name]

    async def score_pair(index_a, item_a, index_b, item_b, left_embeddings, right_embeddings):
        if item_a[5] is None or item_b[5] is None:
            return None
        if embedder is None:
            comparison = await _cpu(request, compare_signatures, item_a[4], item_b[4], s, None)
            if comparison.minimum_pair_score is None:
                return None
            return comparison.minimum_pair_score, comparison.decision
        if embedder.name == "siamese":
            distance = float(np.linalg.norm(left_embeddings[index_a] - right_embeddings[index_b]))
            score = round(float(np.clip(1.0 - distance / 2.0, 0.0, 1.0)), 4)
        else:
            score = round(float(np.clip(np.dot(left_embeddings[index_a], right_embeddings[index_b]), 0.0, 1.0)), 4)
        return score, decide(score, match_at, review_at)

    async def within_pairs(items, embeddings):
        pairs = []
        for index_a, item_a in enumerate(items):
            for index_b in range(index_a + 1, len(items)):
                item_b = items[index_b]
                scored = await score_pair(index_a, item_a, index_b, item_b, embeddings, embeddings)
                if scored is None:
                    continue
                score, decision = scored
                pairs.append(SignaturePairResult(
                    signatures=[index_a + 1, index_b + 1],
                    pages=[item_a[2], item_b[2]],
                    score=score, decision=decision,
                ))
        return pairs

    within_a = await within_pairs(prepared_a, embeddings_a)
    within_b = await within_pairs(prepared_b, embeddings_b)

    def document_result(filename, candidates, comparisons):
        page_numbers = sorted({candidate[2] for candidate in candidates})
        page_number = page_numbers[0] if len(page_numbers) == 1 else None
        return DocumentSignatureResult(
            filename=filename or "uploaded-document",
            page=page_number,
            decision=combine(pair.decision for pair in comparisons) if comparisons else "unavailable",
            comparisons=comparisons,
        )

    documents = [document_result(filename_a, prepared_a, within_a)]
    if pages_b is not None:
        documents.append(document_result(filename_b, prepared_b, within_b))
    signature_decision = combine(document.decision for document in documents)
    within_scores = [pair.score for document in documents for pair in document.comparisons]
    signature_score = round(min(within_scores), 4) if within_scores else None
    if not within_scores:
        warnings.append("At least two usable signatures are required within a document to compare consistency.")
    method = embedder.name if embedder is not None else "heuristic"
    return SignatureResult(
        decision=signature_decision,
        minimum_pair_score=signature_score,
        method=method,
        documents=documents,
        warnings=[*warnings,
                  "Signatures are compared only within each document; no cross-document signature scores are produced."],
    )


async def _face_check_pages(request: Request, pages_a: list[np.ndarray], pages_b: list[np.ndarray] | None,
                            requested_page_a: int, requested_page_b: int,
                            box_a: Box | None = None, box_b: Box | None = None,
                            filename_a: str = "document A", filename_b: str = "document B") -> FaceResult:
    faces_a = []
    faces_b = []
    for page_number, image in enumerate(pages_a):
        face_image = _crop(image, box_a, "face_box_a")
        for candidate in await _cpu(request, request.app.state.faces.candidates, face_image):
            if box_a is not None:
                x0, y0, x1, y1 = box_a
                cx0, cy0, cx1, cy1 = candidate.box
                candidate = type(candidate)(candidate.embedding,
                    (x0 + cx0 * (x1 - x0), y0 + cy0 * (y1 - y0),
                     x0 + cx1 * (x1 - x0), y0 + cy1 * (y1 - y0)))
            faces_a.append((page_number, candidate))
    for page_number, image in enumerate(pages_b or []):
        face_image = _crop(image, box_b, "face_box_b")
        for candidate in await _cpu(request, request.app.state.faces.candidates, face_image):
            if box_b is not None:
                x0, y0, x1, y1 = box_b
                cx0, cy0, cx1, cy1 = candidate.box
                candidate = type(candidate)(candidate.embedding,
                    (x0 + cx0 * (x1 - x0), y0 + cy0 * (y1 - y0),
                     x0 + cx1 * (x1 - x0), y0 + cy1 * (y1 - y0)))
            faces_b.append((page_number, candidate))

    count_a, count_b = len(faces_a), len(faces_b)
    faces_a.sort(key=lambda item: (item[0], (item[1].box[1] + item[1].box[3]) / 2,
                                  (item[1].box[0] + item[1].box[2]) / 2))
    faces_b.sort(key=lambda item: (item[0], (item[1].box[1] + item[1].box[3]) / 2,
                                  (item[1].box[0] + item[1].box[2]) / 2))
    truncated = count_a > MAX_CANDIDATES_PER_DOCUMENT or count_b > MAX_CANDIDATES_PER_DOCUMENT
    faces_a = faces_a[:MAX_CANDIDATES_PER_DOCUMENT]
    faces_b = faces_b[:MAX_CANDIDATES_PER_DOCUMENT]
    warnings = []
    if count_a == 0:
        warnings.append(f"No face detected in document A across {len(pages_a)} page(s).")
    if pages_b is not None and count_b == 0:
        warnings.append(f"No face detected in document B across {len(pages_b)} page(s).")
    if truncated:
        warnings.append(f"Candidate comparisons are limited to the largest {MAX_CANDIDATES_PER_DOCUMENT} faces per document.")

    def within_pairs(faces, requested_page):
        pairs = []
        for candidate_a, (page_a, face_a) in enumerate(faces):
            for candidate_b in range(candidate_a + 1, len(faces)):
                page_b, face_b = faces[candidate_b]
                similarity = float(np.clip(np.dot(face_a.embedding, face_b.embedding), -1.0, 1.0))
                pairs.append(FaceCandidateComparison(
                    candidate_a=candidate_a, candidate_b=candidate_b,
                    page_a=page_a if requested_page == -1 else requested_page,
                    page_b=page_b if requested_page == -1 else requested_page,
                    box_a=list(face_a.box), box_b=list(face_b.box),
                    similarity=round(similarity, 4),
                    decision=request.app.state.faces.decide_similarity(similarity),
                ))
        return pairs

    within_a = within_pairs(faces_a, requested_page_a)
    within_b = within_pairs(faces_b, requested_page_b)

    def document_consistency(filename, faces, pair_results):
        return FaceDocumentConsistency(
            filename=filename,
            faces_found=len(faces),
            decision=combine(pair.decision for pair in pair_results) if pair_results else "unavailable",
            comparisons=pair_results,
        )

    within_documents = [document_consistency(filename_a, faces_a, within_a)]
    if pages_b is not None:
        within_documents.append(document_consistency(filename_b, faces_b, within_b))
    within_decision = combine(item.decision for item in within_documents)
    if not faces_a or pages_b is None or not faces_b:
        return FaceResult(decision="unavailable", similarity=None, faces_found_a=count_a,
                          faces_found_b=count_b, within_document_a=within_a,
                          within_document_b=within_b, within_document_decision=within_decision,
                          within_documents=within_documents, warnings=warnings)

    comparisons = []
    for candidate_a, (page_a, face_a) in enumerate(faces_a):
        for candidate_b, (page_b, face_b) in enumerate(faces_b):
            similarity = float(np.clip(np.dot(face_a.embedding, face_b.embedding), -1.0, 1.0))
            comparisons.append(FaceCandidateComparison(
                candidate_a=candidate_a, candidate_b=candidate_b,
                page_a=page_a if requested_page_a == -1 else requested_page_a,
                page_b=page_b if requested_page_b == -1 else requested_page_b,
                box_a=list(face_a.box), box_b=list(face_b.box), similarity=round(similarity, 4),
                decision=request.app.state.faces.decide_similarity(similarity),
            ))
    best = max(comparisons, key=lambda item: item.similarity)
    if count_a > 1:
        warnings.append(f"Compared {count_a} face candidates in document A against {count_b} in document B.")
    return FaceResult(
        decision=best.decision, similarity=best.similarity,
        faces_found_a=count_a, faces_found_b=count_b,
        page_a=best.page_a, page_b=best.page_b,
        candidate_comparisons=comparisons,
        within_document_a=within_a,
        within_document_b=within_b,
        within_document_decision=within_decision,
        within_documents=within_documents,
        warnings=[*warnings, f"Best face pair is candidate {best.candidate_a} and candidate {best.candidate_b} "
                  f"from pages {best.page_a} and {best.page_b}."],
    )

# --------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def upload_page() -> str:
        return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DocVerify | Document comparison</title>
    <style>
        :root { color-scheme: light; --ink: #1e2925; --muted: #66716a; --paper: #f3f4ef; --panel: #fffefa; --line: #d9ded6; --green: #176b50; }
        * { box-sizing: border-box; }
        body { margin: 0; min-height: 100vh; color: var(--ink); background: radial-gradient(ellipse at 90% 5%, #dce9dd 0, transparent 34%), var(--paper); font: 16px/1.5 ui-sans-serif, system-ui, sans-serif; }
        header { border-bottom: 1px solid var(--line); background: #fbfbf7d9; }
        .top, main { width: min(1040px, calc(100% - 40px)); margin: auto; }
        .top { min-height: 70px; display: flex; align-items: center; justify-content: space-between; }
        .brand { font: 700 22px/1 Georgia, serif; letter-spacing: .01em; }
        .tag { color: var(--muted); font-size: 13px; }
        main { padding: 56px 0 72px; }
        h1 { margin: 0; font: 500 clamp(34px, 5vw, 54px)/1.05 Georgia, serif; letter-spacing: 0; }
        .intro { color: var(--muted); max-width: 620px; margin: 14px 0 32px; }
        .layout { display: grid; grid-template-columns: minmax(280px, .9fr) minmax(340px, 1.1fr); gap: 22px; align-items: start; }
        section { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 24px; }
        h2 { margin: 0 0 18px; font: 600 20px/1.2 Georgia, serif; }
        .drop { display: grid; gap: 10px; padding: 26px 18px; text-align: center; border: 1px dashed #a8b5a8; border-radius: 5px; background: #f8faf5; cursor: pointer; }
        .drop strong { font-size: 15px; }
        .drop span, .hint { color: var(--muted); font-size: 13px; }
        input[type=file] { position: absolute; width: 1px; height: 1px; opacity: 0; }
        .files { list-style: none; padding: 0; margin: 14px 0 20px; display: grid; gap: 7px; }
        .files li { display: flex; justify-content: space-between; gap: 12px; padding: 8px 10px; background: #f3f5ef; border-radius: 3px; overflow-wrap: anywhere; font-size: 13px; }
        button { width: 100%; padding: 12px 16px; color: white; background: var(--green); border: 0; border-radius: 4px; font: 650 15px/1.2 ui-sans-serif, system-ui, sans-serif; cursor: pointer; }
        button:hover { background: #10563f; }
        button:disabled { opacity: .55; cursor: wait; }
        .status { min-height: 24px; margin: 11px 0 0; color: var(--muted); font-size: 13px; }
        pre { min-height: 260px; max-height: 620px; overflow: auto; margin: 0; padding: 16px; color: #e5eee7; background: #202a26; border-radius: 4px; font: 13px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
        .result-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
        .result-head h2 { margin-bottom: 14px; }
        .result-head span { color: var(--muted); font-size: 12px; }
        .result-summary { display: grid; gap: 18px; }
        .result-section { border-top: 1px solid var(--line); padding-top: 14px; }
        .result-section h3 { margin: 0 0 10px; font: 600 16px/1.3 Georgia, serif; }
        .result-line { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; font-size: 14px; }
        .result-line span:first-child, .muted { color: var(--muted); }
        .decision { font-weight: 700; text-transform: capitalize; }
        .pair-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .pair-table th, .pair-table td { padding: 7px 5px; border-bottom: 1px solid var(--line); text-align: left; }
        .pair-table th { color: var(--muted); font-weight: 600; }
        details { margin-top: 14px; }
        details summary { color: var(--muted); font-size: 12px; cursor: pointer; }
        details pre { margin-top: 8px; }
        .privacy { margin-top: 18px; color: var(--muted); font-size: 12px; }
        @media (max-width: 760px) { .top, main { width: min(100% - 28px, 560px); } main { padding-top: 38px; } .layout { grid-template-columns: 1fr; } section { padding: 18px; } }
    </style>
</head>
<body>
    <header><div class="top"><div class="brand">DocVerify</div><div class="tag">Document identity comparison</div></div></header>
    <main>
        <h1>Compare documents</h1>
        <p class="intro">Upload one application form to check consistency among its specimen signatures, or two documents to compare faces.</p>
        <div class="layout">
            <section>
                <h2>Documents</h2>
                <label class="drop" for="documents"><strong>Select document files</strong><span>One file checks internal signatures; two files also compare faces</span></label>
                <input id="documents" type="file" accept="image/*,.pdf,application/pdf" multiple>
                <ul class="files" id="file-list"><li class="hint">No files selected</li></ul>
                <button id="compare" type="button">Compare documents</button>
                <p class="status" id="status" role="status" aria-live="polite"></p>
                <p class="privacy">Files are processed in memory and are not saved by the service. Results support human review and are not forensic or legal proof.</p>
            </section>
            <section>
                <div class="result-head"><h2>Comparison result</h2><span id="route-label">Waiting for upload</span></div>
                <div id="result" aria-live="polite" class="result-summary"><p class="muted">Upload one form for signature consistency or two files for face comparison.</p></div>
            </section>
        </div>
    </main>
    <script>
        const picker = document.querySelector('#documents');
        const list = document.querySelector('#file-list');
        const button = document.querySelector('#compare');
        const status = document.querySelector('#status');
        const result = document.querySelector('#result');
        const routeLabel = document.querySelector('#route-label');
        const renderResult = data => {
            result.replaceChildren();
            const addLine = (parent, label, value, className = '') => {
                const row = document.createElement('div'); row.className = 'result-line';
                const name = document.createElement('span'); name.textContent = label;
                const content = document.createElement('span'); content.textContent = value; content.className = className;
                row.append(name, content); parent.append(row);
            };
            addLine(result, 'Overall', data.overall || 'unavailable', 'decision');
            const faceSection = document.createElement('section'); faceSection.className = 'result-section';
            const faceTitle = document.createElement('h3'); faceTitle.textContent = 'Face comparison'; faceSection.append(faceTitle);
            addLine(faceSection, 'Decision', data.face?.decision || 'unavailable', 'decision');
            if (data.face?.similarity !== null && data.face?.similarity !== undefined) addLine(faceSection, 'Similarity', data.face.similarity.toFixed(4));
            if (data.face) result.append(faceSection);
            const faceDocuments = data.face_consistency || data.face?.within_documents || [];
            if (faceDocuments.length) {
                const internalFaceSection = document.createElement('section'); internalFaceSection.className = 'result-section';
                const internalFaceTitle = document.createElement('h3'); internalFaceTitle.textContent = 'Within-document face consistency'; internalFaceSection.append(internalFaceTitle);
                for (const file of faceDocuments) {
                    const fileTitle = document.createElement('h3'); fileTitle.textContent = file.filename;
                    internalFaceSection.append(fileTitle);
                    addLine(internalFaceSection, 'Faces detected', String(file.faces_found));
                    addLine(internalFaceSection, 'Consistency', file.decision || 'unavailable', 'decision');
                    if (file.comparisons?.length) {
                        const table = document.createElement('table'); table.className = 'pair-table';
                        table.innerHTML = '<thead><tr><th>Faces</th><th>Pages</th><th>Similarity</th><th>Result</th></tr></thead>';
                        const body = document.createElement('tbody');
                        for (const pair of file.comparisons) {
                            const row = document.createElement('tr');
                            const pages = pair.page_a === pair.page_b ? String(pair.page_a + 1) : `${pair.page_a + 1} & ${pair.page_b + 1}`;
                            for (const value of [`${pair.candidate_a + 1} & ${pair.candidate_b + 1}`, pages, Number(pair.similarity).toFixed(4), pair.decision || 'unavailable']) {
                                const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
                            }
                            body.append(row);
                        }
                        table.append(body); internalFaceSection.append(table);
                    }
                }
                result.append(internalFaceSection);
            }
            if (data.comparisons?.length) {
                const pairSection = document.createElement('section'); pairSection.className = 'result-section';
                const pairTitle = document.createElement('h3'); pairTitle.textContent = 'Face comparisons'; pairSection.append(pairTitle);
                const table = document.createElement('table'); table.className = 'pair-table';
                table.innerHTML = '<thead><tr><th>Documents</th><th>Similarity</th><th>Result</th></tr></thead>';
                const body = document.createElement('tbody');
                const filenames = (data.signature_consistency?.documents || []).map(file => file.filename);
                for (const pair of data.comparisons) {
                    const row = document.createElement('tr');
                    const documentLabels = filenames.length
                        ? `${filenames[pair.document_a]} & ${filenames[pair.document_b]}`
                        : `${pair.document_a + 1} & ${pair.document_b + 1}`;
                    for (const value of [documentLabels, pair.face?.similarity == null ? '—' : Number(pair.face.similarity).toFixed(4), pair.face?.decision || 'unavailable']) {
                        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
                    }
                    body.append(row);
                }
                table.append(body); pairSection.append(table); result.append(pairSection);
            }
            const signature = data.signature || data.signature_consistency;
            if (signature) {
                const signatureSection = document.createElement('section'); signatureSection.className = 'result-section';
                const title = document.createElement('h3'); title.textContent = 'Specimen signature consistency'; signatureSection.append(title);
                addLine(signatureSection, 'Decision', signature.decision || 'unavailable', 'decision');
                addLine(signatureSection, 'Scoring model', signature.method || 'unknown');
                for (const file of signature.documents || []) {
                    const fileTitle = document.createElement('h3');
                    fileTitle.textContent = `${file.filename} · ${file.page === null ? 'multiple pages' : `page ${file.page + 1}`}`;
                    signatureSection.append(fileTitle);
                    addLine(signatureSection, 'Form consistency', file.decision || 'unavailable', 'decision');
                    const table = document.createElement('table'); table.className = 'pair-table';
                    table.innerHTML = '<thead><tr><th>Slots</th><th>Score</th><th>Result</th></tr></thead>';
                    const body = document.createElement('tbody');
                    for (const pair of file.comparisons || []) {
                        const row = document.createElement('tr');
                        for (const value of [`${pair.signatures?.join(' & ') || 'Pair'}`, Number(pair.score).toFixed(4), pair.decision || 'unavailable']) {
                            const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
                        }
                        body.append(row);
                    }
                    table.append(body); signatureSection.append(table);
                }
                for (const warning of signature.warnings || []) {
                    const note = document.createElement('p'); note.className = 'muted'; note.textContent = warning; signatureSection.append(note);
                }
                result.append(signatureSection);
            }
            const details = document.createElement('details');
            const summary = document.createElement('summary'); summary.textContent = 'API details';
            const raw = document.createElement('pre'); raw.textContent = JSON.stringify(data, null, 2);
            details.append(summary, raw); result.append(details);
        };
        picker.addEventListener('change', () => {
            list.replaceChildren();
            if (!picker.files.length) {
                const item = document.createElement('li'); item.className = 'hint'; item.textContent = 'No files selected'; list.append(item); return;
            }
            for (const file of picker.files) {
                const item = document.createElement('li');
                const name = document.createElement('span'); name.textContent = file.name;
                const size = document.createElement('span'); size.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
                item.append(name, size); list.append(item);
            }
            status.textContent = `${picker.files.length} document${picker.files.length === 1 ? '' : 's'} selected.`;
        });
        button.addEventListener('click', async () => {
            const files = [...picker.files];
            if (files.length < 1) { status.textContent = 'Choose a document.'; return; }
            const multiple = files.length > 2;
            const form = new FormData();
            if (multiple) {
                files.forEach(file => { form.append('documents', file); form.append('pages', '-1'); });
            } else {
                form.append('doc_a', files[0]); form.append('page_a', '-1');
                if (files.length === 2) {
                    form.append('doc_b', files[1]); form.append('page_b', '-1');
                }
            }
            const endpoint = multiple ? '/v1/compare/multiple' : '/v1/compare';
            button.disabled = true; status.textContent = 'Comparing documents…'; routeLabel.textContent = endpoint;
            result.textContent = 'Processing…';
            try {
                const response = await fetch(endpoint, { method: 'POST', body: form });
                const data = await response.json();
                renderResult(data);
                status.textContent = response.ok ? 'Comparison complete.' : `Request failed (${response.status}). See the response details.`;
            } catch (error) {
                renderResult({ overall: 'unavailable', error: String(error) });
                status.textContent = 'Could not reach the DocVerify service.';
            } finally { button.disabled = false; }
        });
    </script>
</body>
</html>"""

@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/v1/compare", response_model=CompareResult, tags=["compare"])
async def compare_documents(
    request: Request,
    s: SettingsDep,
    doc_a: UploadFile = File(...),
    doc_b: UploadFile | None = File(None),
    sig_box_a: str | None = Form(None, description="Signature region in doc A: 'x0,y0,x1,y1' (0..1)"),
    sig_box_b: str | None = Form(None, description="Signature region in doc B"),
    page_a: int = Form(0, ge=-1),
    page_b: int = Form(0, ge=-1),
) -> CompareResult:
    """Face comparison always runs. Signature comparison runs if BOTH boxes are provided, or, when a
    signature detector is configured, if neither is (the detector then locates both signatures)."""
    ba, bb = _parse_box(sig_box_a, "sig_box_a"), _parse_box(sig_box_b, "sig_box_b")
    if (ba is None) != (bb is None):
        raise HTTPException(422, "Provide both sig_box_a and sig_box_b, or neither.")
    if (page_a == -1 or page_b == -1) and ba is not None:
        raise HTTPException(422, "Manual signature boxes require a single selected page.")

    pages_a = await _load_pages(request, doc_a, page_a, s)
    pages_b = await _load_pages(request, doc_b, page_b, s) if doc_b is not None else None
    if pages_b is None:
        face = await _face_check_pages(request, pages_a, None, page_a, page_b,
                                       filename_a=doc_a.filename)
        face.warnings.append("Cross-document face comparison requires a second document.")
    else:
        face = await _face_check_pages(request, pages_a, pages_b, page_a, page_b,
                                       filename_a=doc_a.filename, filename_b=doc_b.filename)

    signature: SignatureResult | None = None
    auto = ba is None and bb is None
    if auto:
        signature = await _signature_check_pages(request, s, pages_a, pages_b,
                                                 doc_a.filename, doc_b.filename if doc_b else None)
    elif ba and bb:
        signature = SignatureResult(
            decision="unavailable",
            warnings=["Single manual boxes cannot measure within-document consistency; configure the signature detector to locate multiple signatures."],
        )
    else:
        signature = SignatureResult(
            decision="unavailable",
            warnings=["Signature consistency was not run: configure specimen slots or a signature detector."],
        )

    if pages_b is None:
        face_decisions = [item.decision for item in face.within_documents if item.comparisons]
        overall = combine([signature.decision, *face_decisions])
    else:
        overall = combine([face.decision, signature.decision, face.within_document_decision])
    result = CompareResult(
        request_id=uuid.uuid4().hex,
        overall=overall,
        face=face,
        signature=signature,
    )
    #Log decisions only: never log or persist uploaded biometric data.
    log.info("compare %s overall=%s face=%s sig=%s", result.request_id, result.overall,
             face.decision, signature.decision if signature else "skipped")
    return result


@app.post("/v1/compare/multiple", response_model=MultiCompareResult, tags=["compare"])
async def compare_multiple_documents(
    request: Request,
    s: SettingsDep,
    documents: list[UploadFile] = File(..., description="Two or more documents; every unique pair is compared."),
    pages: list[int] | None = Form(None, description="Optional zero-based page index per document; use -1 to search all pages."),
    face_boxes: list[str] | None = Form(None, description="Optional relative face crop per document; use an empty value for automatic largest-face selection."),
    sig_boxes: list[str] | None = Form(None, description="Optional relative signature crop per document; empty values use the detector when configured."),
) -> MultiCompareResult:
    count = len(documents)
    if count < 2:
        raise HTTPException(422, "Upload at least two documents.")
    if count > s.max_compare_documents:
        raise HTTPException(422, f"At most {s.max_compare_documents} documents may be compared at once.")

    page_values = pages or [0] * count
    if len(page_values) != count or any(page < -1 for page in page_values):
        raise HTTPException(422, "Provide one non-negative page index per document, or omit pages.")

    def parse_boxes(values: list[str] | None, field: str) -> list[Box | None]:
        if values is None:
            return [None] * count
        if len(values) != count:
            raise HTTPException(422, f"Provide one {field} value per document, or omit {field}.")
        return [_parse_box(value, f"{field}[{index}]") for index, value in enumerate(values)]

    face_regions = parse_boxes(face_boxes, "face_boxes")
    signature_regions = parse_boxes(sig_boxes, "sig_boxes")
    if any(signature_regions):
        raise HTTPException(422, "Single signature boxes cannot measure within-document consistency; use configured specimen slots or automatic detection.")
    images = [await _load_pages(request, upload, page, s) for upload, page in zip(documents, page_values)]

    consistency_results = [
        await _signature_check_pages(request, s, image_pages, None, upload.filename)
        for upload, image_pages in zip(documents, images)
    ]
    signature_consistency = SignatureResult(
        decision=combine(result.decision for result in consistency_results),
        minimum_pair_score=min(
            (result.minimum_pair_score for result in consistency_results
             if result.minimum_pair_score is not None),
            default=None,
        ),
        method=consistency_results[0].method,
        documents=[document for result in consistency_results for document in result.documents],
        warnings=list(dict.fromkeys(warning for result in consistency_results for warning in result.warnings)),
    )
    face_consistency_results = [
        await _face_check_pages(request, image_pages, None, page_values[index], -1,
                                filename_a=documents[index].filename)
        for index, image_pages in enumerate(images)
    ]
    face_consistency = [result.within_documents[0] for result in face_consistency_results]
    face_file_decisions = [item.decision for item in face_consistency]

    comparisons: list[PairCompareResult] = []
    for index_a, index_b in combinations(range(count), 2):
        face = await _face_check_pages(request, images[index_a], images[index_b],
                           page_values[index_a], page_values[index_b],
                           face_regions[index_a], face_regions[index_b],
                           documents[index_a].filename, documents[index_b].filename)

        file_decisions = [result.documents[0].decision if result.documents else "unavailable"
                          for result in consistency_results]

        comparisons.append(PairCompareResult(
            document_a=index_a,
            document_b=index_b,
            overall=combine([face.decision, face_file_decisions[index_a], face_file_decisions[index_b],
                             file_decisions[index_a], file_decisions[index_b]]),
            face=face,
        ))

    overall = combine([*(pair.overall for pair in comparisons), signature_consistency.decision,
                        *(item.decision for item in face_consistency)])
    result = MultiCompareResult(
        request_id=uuid.uuid4().hex,
        document_count=count,
        overall=overall,
        comparisons=comparisons,
        face_consistency=face_consistency,
        signature_consistency=signature_consistency,
    )
    log.info("compare %s documents=%s pairs=%s overall=%s", result.request_id, count, len(comparisons), overall)
    return result
