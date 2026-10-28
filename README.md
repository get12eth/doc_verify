# DocVerify

FastAPI service for comparing portrait photos and handwritten signatures across document images or PDFs. Returns structured JSON with similarity scores, evidence availability, warnings, and a decision: `match`, `review`, `no_match`, or `unavailable`.


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

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/compare` | Compare two documents |
| `POST` | `/v1/compare/multiple` | Compare every unique pair |


### Multiple documents

 All unique pairs are compared; max 10 documents by default.


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
  "disclaimer": "Automated similarity score for decision support only;
}
```

## Signature models

Automatic page detection with a YOLOv8 ONNX detector
