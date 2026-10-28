# DocVerify

FastAPI service for comparing portrait photos and handwritten signatures across document images or PDFs. Returns structured JSON with similarity scores, evidence availability, warnings, and a decision: `match`, `review`, `no_match`, or `unavailable`.

Result is decision support only — not forensic or legal proof.

## Requirements

- Python 3.10+
- Tesseract OCR on `PATH` (specimen-signature row detection)
- Internet on first startup to download the InsightFace `buffalo_l` model pack (unless cached)

## Install and run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows: `venv\Scripts\activate`. Use one Uvicorn worker.

Open `http://127.0.0.1:8000/` (UI) or `/docs` (API docs).

## API

Both endpoints accept multipart uploads (images or PDF; PDF page indexes are zero-based).

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/compare` | Compare two documents |
| `POST` | `/v1/compare/multiple` | Compare every unique pair |

### Two documents

Required: `doc_a`, `doc_b`. Optional: `page_a`, `page_b`, `sig_box_a`, `sig_box_b`.

Signature boxes: normalized `x0,y0,x1,y1` (0–1). Supply both or neither.

```bash
curl -X POST http://127.0.0.1:8000/v1/compare \
  -F 'doc_a=@id_card.jpg' \
  -F 'doc_b=@contract.pdf' \
  -F 'page_b=0' \
  -F 'sig_box_a=0.55,0.80,0.95,0.95' \
  -F 'sig_box_b=0.10,0.85,0.45,0.97'
```

Face comparison always runs. Signature comparison runs when both boxes are supplied or a signature detector is configured; otherwise the signature decision is `unavailable`.

Omit `doc_b` for within-document signature consistency only (face comparison requires two files).

### Multiple documents

Pass each file as a `documents` field. All unique pairs are compared; max 10 documents by default.

```bash
curl -X POST http://127.0.0.1:8000/v1/compare/multiple \
  -F 'documents=@id_card.jpg' \
  -F 'documents=@contract.pdf'
```

Optional per-document fields: `pages`, `face_boxes`, `sig_boxes`. Use `-1` to search every PDF page. Empty box values request automatic detection when a detector is configured.

### Response

```json
{
  "request_id": "...",
  "overall": "review",
  "face": {
    "decision": "match",
    "similarity": 0.82,
    "faces_found_a": 1,
    "faces_found_b": 1,
    "warnings": []
  },
  "signature": {
    "decision": "unavailable",
    "score": null,
    "method": "heuristic",
    "components": {},
    "box_source": "none",
    "box_a": null,
    "box_b": null,
    "warnings": []
  },
  "disclaimer": "Automated similarity score for decision support only; not forensic or legal proof."
}
```

## Signature models

Built-in image-analysis heuristic (requires `sig_box_a` and `sig_box_b`).

Automatic page detection with a YOLOv8 ONNX detector:

```bash
export DOCVERIFY_SIG_DETECTOR_MODEL=models/signature_detector.onnx
export DOCVERIFY_SIG_DETECTOR_MIN_Y=0.88
```

SigVerse verifier:

```bash
pip install sigverse
export DOCVERIFY_SIG_SIGVERSE=true
```

Trained local Siamese model:

```bash
export DOCVERIFY_SIG_SIAMESE_MODEL=models/signature_siamese.onnx
```

Use at most one optional signature model. Calibrate thresholds on representative documents before relying on decisions.

## Train a Siamese model

Requires ≥4 writers with ≥2 genuine signatures each:

```
DATA/
  writer_001/
    genuine/  signature_01.png
    forged/   forgery_01.png   # optional
```

```bash
./venv/bin/pip install -r requirements-train.txt
./venv/bin/python -m scripts.train_siamese --data DATA --out models/signature_siamese.onnx --epochs 30
```

The trainer holds out writers for validation and prints the validation EER.

## Configuration

Environment variables (`DOCVERIFY_*`) or `.env`:

| Setting | Default |
|---------|---------|
| `DOCVERIFY_MAX_UPLOAD_MB` | 15 |
| `DOCVERIFY_MAX_COMPARE_DOCUMENTS` | 10 |
| `DOCVERIFY_PDF_DPI` | 300 |
| `DOCVERIFY_MAX_CONCURRENCY` | 2 |
| `DOCVERIFY_FACE_MATCH_THRESHOLD` | 0.45 |
| `DOCVERIFY_FACE_REVIEW_THRESHOLD` | 0.30 |

Configure your reverse proxy's request-size limit too.

## Deployment

Uploads are processed in memory and not intentionally stored. Treat document photos and signatures as sensitive biometric data. InsightFace's model pack has non-commercial licensing restrictions; check its terms before commercial use.

Serve behind TLS and a reverse proxy. Do not expose the development server to the public internet.
