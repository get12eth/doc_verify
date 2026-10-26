# DocVerify

DocVerify is a FastAPI service for comparing portrait photos and handwritten signatures across document images or PDFs. It returns structured JSON with similarity scores, evidence availability, warnings, and a decision: `match`, `review`, `no_match`, or `unavailable`.

The result is decision support only. It is not forensic or legal proof.

## Requirements

- Python 3.10 or newer
- Linux, macOS, or Windows
- Internet access on first startup to download the InsightFace `buffalo_l` model pack, unless it is already cached

## Install and run

From the project directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On Windows, activate the environment with `venv\\Scripts\\activate`. Keep Uvicorn at one worker; each worker loads its own models.

Open the upload page at <http://127.0.0.1:8000/>. The interactive API documentation is at <http://127.0.0.1:8000/docs>.

## Upload page

Select two or more image or PDF files and choose **Compare documents**. Two files are sent to `POST /v1/compare`; three or more are sent to `POST /v1/compare/multiple`. The page displays the JSON returned by the API.

## API endpoints

There are two document-comparison endpoints:

| Method and path | Purpose |
|---|---|
| `POST /v1/compare` | Compare two documents |
| `POST /v1/compare/multiple` | Compare every unique pair in a set of 2 or more documents |

Both endpoints accept multipart form uploads. Supported document types depend on the decoder: common image formats and PDF. PDF page indexes are zero-based.

### Compare two documents

Required fields: `doc_a`, `doc_b`. Optional fields: `page_a`, `page_b`, `sig_box_a`, `sig_box_b`.

Signature boxes are normalized page coordinates in `x0,y0,x1,y1` format, where each value is from 0 to 1. Supply both boxes or neither.

```bash
curl -X POST http://127.0.0.1:8000/v1/compare \
  -F 'doc_a=@id_card.jpg' \
  -F 'doc_b=@contract.pdf' \
  -F 'page_b=0' \
  -F 'sig_box_a=0.55,0.80,0.95,0.95' \
  -F 'sig_box_b=0.10,0.85,0.45,0.97'
```

The face comparison always runs. Signature comparison runs when both signature boxes are supplied or when a signature detector is configured. If neither is available, the signature decision is `unavailable` and the overall decision is `review` unless face evidence is a `no_match`.

### Compare multiple documents

Pass each file as another `documents` field. The service compares all unique pairs; for $N$ documents there are $N(N-1)/2$ comparisons. The default maximum is 10 documents.

```bash
curl -X POST http://127.0.0.1:8000/v1/compare/multiple \
  -F 'documents=@id_card.jpg' \
  -F 'documents=@contract.pdf' \
  -F 'documents=@passport.jpg'
```

Optional repeated fields are `pages`, `face_boxes`, and `sig_boxes`; provide one value per document when using them. Page indexes are zero-based. Boxes use normalized `x0,y0,x1,y1` coordinates. Leave a box value empty to request automatic detection when a signature detector is configured. Pair indexes in the JSON response are zero-based.

### Response format

Responses include a `request_id`, an overall decision, and the per-check results. The multiple-document response contains one result for each document pair. Face results report similarity, the number of detected faces, and warnings. Signature results report similarity, scoring method, box information, and warnings. A score is not a probability.

Typical pair response shape:

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

## Signature comparison options

Without optional models, signatures can be compared using the built-in image-analysis heuristic. It requires signature crops via `sig_box_a` and `sig_box_b`.

To detect signatures on full document pages automatically, provide a compatible single-class YOLOv8 signature detector exported to ONNX:

```bash
export DOCVERIFY_SIG_DETECTOR_MODEL=models/signature_detector.onnx
```

Check the detector's license and output format before use. The detector must return the standard YOLOv8 output shape `(1, 4+nc, N)`.
For forms with signatures in a known page region, `DOCVERIFY_SIG_DETECTOR_MIN_Y` filters out boxes above that normalized vertical position (default `0.88`, meaning the bottom 12% of the page). Set it to `0` to search the full page; tune it for other document layouts.

Alternatively, enable the pretrained SigVerse verifier:

```bash
pip install sigverse
export DOCVERIFY_SIG_SIGVERSE=true
```

Or use a trained local Siamese model by setting `DOCVERIFY_SIG_SIAMESE_MODEL`, as described below. Use at most one optional signature model. Calibrate thresholds on representative documents before relying on decisions; scores from different methods are not interchangeable.

## Train a signature model

Training requires at least four writers, with at least two genuine signatures per writer. Optional forged samples provide hard negatives. Arrange files like this:

```text
DATA/
  writer_001/
    genuine/
      signature_01.png
      signature_02.png
    forged/                 # optional
      forgery_01.png
  writer_002/
    genuine/
      ...
```

If the pre-downloaded dataset already has this layout, use it directly. Otherwise, `dataset.py` copies from a source directory with the same writer/genuine/forged subfolder structure:

```bash
./venv/bin/python dataset.py --source /path/to/local_dataset --out DATA
```

Install training dependencies, train, then point the application at the exported ONNX model:

```bash
./venv/bin/pip install -r requirements-train.txt
./venv/bin/python -m scripts.train_siamese \
  --data DATA \
  --out models/signature_siamese.onnx \
  --epochs 30
export DOCVERIFY_SIG_SIAMESE_MODEL=models/signature_siamese.onnx
```

The trainer holds out writers for validation and prints the validation equal error rate (EER). The API and trainer share signature preprocessing. The Siamese model uses Euclidean embedding distance, mapped to a 0-to-1 similarity score by `1 - distance / 2`. Recalibrate thresholds on your own data after training.

## Configuration and limits

Settings can be supplied as environment variables prefixed with `DOCVERIFY_` or in a project `.env` file. Common settings:

| Setting | Default | Purpose |
|---|---:|---|
| `DOCVERIFY_MAX_UPLOAD_MB` | `15` | Maximum size of each upload |
| `DOCVERIFY_MAX_COMPARE_DOCUMENTS` | `10` | Maximum files in the multiple-document request |
| `DOCVERIFY_PDF_DPI` | `300` | PDF rendering resolution |
| `DOCVERIFY_MAX_CONCURRENCY` | `2` | Maximum concurrent CPU-bound requests |
| `DOCVERIFY_FACE_MATCH_THRESHOLD` | `0.45` | Initial ArcFace match threshold |
| `DOCVERIFY_FACE_REVIEW_THRESHOLD` | `0.30` | Initial ArcFace review threshold |

Threshold defaults are starting points, not validated operating values. Configure the reverse proxy's request-size limit as well as the application upload limit.

## Privacy and deployment

Uploads are processed in memory and are not intentionally stored or logged. Treat document photos and signatures as sensitive biometric data, and apply the privacy, retention, and access controls required for your use case. InsightFace's pretrained model pack has non-commercial licensing restrictions; check its terms before commercial deployment.

For production, serve behind TLS and a configured reverse proxy. Do not expose the local development server directly to the public internet.