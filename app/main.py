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
from .decision import combine
from .detector import load_detector
from .embedders import load_embedder
from .face import FaceMatcher
from .imaging import InvalidDocument, load_document
from .schemas import CompareResult, MultiCompareResult, PairCompareResult, SignatureResult
from .signature import compare_signatures

log = logging.getLogger("docverify")
T = TypeVar("T")
Box = tuple[float, float, float, float]

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


async def _load(request: Request, upload: UploadFile, page: int, s: Settings) -> np.ndarray:
    data = await _read(upload, s)
    try:
        return await _cpu(request, functools.partial(load_document, data, page=page, dpi=s.pdf_dpi,
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


async def _signature_check(request: Request, s: Settings, img_a: np.ndarray, img_b: np.ndarray,
                           box_a: Box | None, box_b: Box | None, *, auto: bool) -> SignatureResult:
    """Compare signatures; with auto=True, missing boxes are located by the detector."""
    detector = request.app.state.detector
    if auto and detector is None:
        raise HTTPException(422, "No signature detector configured (set DOCVERIFY_SIG_DETECTOR_MODEL).")

    manual = box_a is not None or box_b is not None
    detected = False
    warnings: list[str] = []
    if auto:
        for label, img, box in (("A", img_a, box_a), ("B", img_b, box_b)):
            if box is None:
                detections = await _cpu(request, detector.detect, img)
                if len(detections) > 1:
                    warnings.append(f"{len(detections)} signatures detected in document {label}; the highest-confidence was used.")
                if detections:
                    x0, y0, x1, y1 = detections[0].box
                    px = (x1 - x0) * s.sig_detector_pad
                    py = (y1 - y0) * s.sig_detector_pad
                    box = (max(0.0, x0 - px), max(0.0, y0 - py),
                           min(1.0, x1 + px), min(1.0, y1 + py))
                detected = detected or box is not None
                if box is None:
                    warnings.append(f"No signature detected in document {label}.")
            if label == "A":
                box_a = box
            else:
                box_b = box
        if box_a is None or box_b is None:
            return SignatureResult(decision="unavailable", box_source="auto", warnings=warnings)

    result = await _cpu(request, compare_signatures, _crop(img_a, box_a, "box_a"), _crop(img_b, box_b, "box_b"),
                        s, request.app.state.embedder)
    return result.model_copy(update={
        "box_source": "auto" if detected else ("manual" if manual else "none"),
        "box_a": list(box_a) if box_a else None,
        "box_b": list(box_b) if box_b else None,
        "warnings": [*warnings, *result.warnings],
    })

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
        .privacy { margin-top: 18px; color: var(--muted); font-size: 12px; }
        @media (max-width: 760px) { .top, main { width: min(100% - 28px, 560px); } main { padding-top: 38px; } .layout { grid-template-columns: 1fr; } section { padding: 18px; } }
    </style>
</head>
<body>
    <header><div class="top"><div class="brand">DocVerify</div><div class="tag">Document identity comparison</div></div></header>
    <main>
        <h1>Compare documents</h1>
        <p class="intro">Upload two or more document images or PDFs. DocVerify compares portrait photos and compares signatures when signature regions are supplied or automatic signature detection is configured.</p>
        <div class="layout">
            <section>
                <h2>Documents</h2>
                <label class="drop" for="documents"><strong>Select document files</strong><span>Images or PDFs · choose at least two</span></label>
                <input id="documents" type="file" accept="image/*,.pdf,application/pdf" multiple>
                <ul class="files" id="file-list"><li class="hint">No files selected</li></ul>
                <button id="compare" type="button">Compare documents</button>
                <p class="status" id="status" role="status" aria-live="polite"></p>
                <p class="privacy">Files are processed in memory and are not saved by the service. Results support human review and are not forensic or legal proof.</p>
            </section>
            <section>
                <div class="result-head"><h2>JSON result</h2><span id="route-label">Waiting for upload</span></div>
                <pre id="result" aria-live="polite">{\n  "status": "ready",\n  "message": "Select at least two documents to begin."\n}</pre>
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
            if (files.length < 2) { status.textContent = 'Choose at least two documents.'; return; }
            const multiple = files.length > 2;
            const form = new FormData();
            if (multiple) files.forEach(file => form.append('documents', file));
            else { form.append('doc_a', files[0]); form.append('doc_b', files[1]); }
            const endpoint = multiple ? '/v1/compare/multiple' : '/v1/compare';
            button.disabled = true; status.textContent = 'Comparing documents…'; routeLabel.textContent = endpoint;
            result.textContent = JSON.stringify({ status: 'processing' }, null, 2);
            try {
                const response = await fetch(endpoint, { method: 'POST', body: form });
                const data = await response.json();
                result.textContent = JSON.stringify(data, null, 2);
                status.textContent = response.ok ? 'Comparison complete.' : `Request failed (${response.status}). See the response details.`;
            } catch (error) {
                result.textContent = JSON.stringify({ error: String(error) }, null, 2);
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
    doc_b: UploadFile = File(...),
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

    img_a, img_b = await _load(request, doc_a, page_a, s), await _load(request, doc_b, page_b, s)
    face = await _cpu(request, request.app.state.faces.compare, img_a, img_b)

    signature: SignatureResult | None = None
    auto = ba is None and bb is None and request.app.state.detector is not None
    if auto or (ba and bb):
        signature = await _signature_check(request, s, img_a, img_b, ba, bb, auto=auto)
    else:
        signature = SignatureResult(
            decision="unavailable",
            warnings=["Signature comparison was not run: provide both signature boxes or configure the signature detector."],
        )

    result = CompareResult(
        request_id=uuid.uuid4().hex,
        overall=combine([face.decision, signature.decision]),
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
    pages: list[int] | None = Form(None, description="Optional zero-based page index per document."),
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
    images = [await _load(request, upload, page, s) for upload, page in zip(documents, page_values)]

    comparisons: list[PairCompareResult] = []
    for index_a, index_b in combinations(range(count), 2):
        face_a = _crop(images[index_a], face_regions[index_a], f"face_boxes[{index_a}]")
        face_b = _crop(images[index_b], face_regions[index_b], f"face_boxes[{index_b}]")
        face = await _cpu(request, request.app.state.faces.compare, face_a, face_b)

        box_a, box_b = signature_regions[index_a], signature_regions[index_b]
        detector_available = request.app.state.detector is not None
        if box_a is not None and box_b is not None:
            signature = await _signature_check(request, s, images[index_a], images[index_b], box_a, box_b, auto=False)
        elif detector_available:
            signature = await _signature_check(request, s, images[index_a], images[index_b], box_a, box_b, auto=True)
        else:
            signature = SignatureResult(
                decision="unavailable",
                warnings=["Signature comparison was not run: provide boxes for both documents or configure the signature detector."],
            )

        comparisons.append(PairCompareResult(
            document_a=index_a,
            document_b=index_b,
            overall=combine([face.decision, signature.decision]),
            face=face,
            signature=signature,
        ))

    overall = combine(pair.overall for pair in comparisons)
    result = MultiCompareResult(
        request_id=uuid.uuid4().hex,
        document_count=count,
        overall=overall,
        comparisons=comparisons,
    )
    log.info("compare %s documents=%s pairs=%s overall=%s", result.request_id, count, len(comparisons), overall)
    return result
