# CivicLens ML Service

AI triage for citizen-reported civic issues (potholes, garbage, damaged streetlights, waterlogging).
A standalone **FastAPI** service that the Node/React backend calls over HTTP.

## Project structure
```
civiclens-ml/
├── app/
│   ├── main.py              # FastAPI app + all endpoints
│   ├── pipeline.py          # orchestrates the full triage flow
│   ├── schemas.py           # request/response contract (shared with backend)
│   ├── config.py            # env-driven settings
│   ├── constants.py         # classes, departments, base severity (edit to add classes)
│   ├── static/demo.html     # standalone demo UI at /demo
│   └── services/
│       ├── exif_validator.py   # A: fake / stale / wrong-location photo checks
│       ├── detector.py         # B: YOLOv8 detector (+ mock fallback for demos)
│       ├── severity.py         # C: explainable severity 1-5
│       ├── priority.py         #    priority score using nearby hospitals/schools/roads
│       ├── clustering.py       # D: DBSCAN duplicate merging
│       ├── eta_model.py        # E: repair-time prediction (RandomForest)
│       ├── materials.py        #    materials / tools / crew lookup
│       ├── privacy.py          #    face + number-plate blur
│       ├── trust.py            #    dynamic reporter trust score
│       ├── hotspot.py          #    hotspot heatmap + 7-day forecast
│       ├── complaint_nlp.py    #    English/Tamil/Hindi complaint classifier
│       └── image_io.py, annotate.py, geo.py, timeutil.py
├── training/                # merge_datasets.py, train_yolo.py, train_eta_model.py, data.yaml, colab_train.md
├── integration/             # node_client.js (for the backend), ai_triage.sql (DB table)
├── tests/                   # 21 pytest tests
├── models/                  # put best.pt here (eta_model.joblib is created automatically)
├── Dockerfile, requirements*.txt, .env.example, MODEL_CARD.md
```

## Run it (3 commands)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open **http://localhost:8000/demo** (upload a photo) or **http://localhost:8000/docs** (interactive API docs).

Without trained weights the service runs in **mock mode**: the whole pipeline works (so your friend can
integrate immediately) but the "detection" is a deterministic placeholder. Every response carries
`"model_mode": "mock"` or `"yolo"`. **Never present mock results as real detection.**

Run the tests: `pip install -r requirements-dev.txt && pytest -q`

## Switch to the real YOLO model
1. Collect data and train (Colab/Kaggle GPU): see `training/colab_train.md`
   `merge_datasets.py` -> `train_yolo.py` produces `models/best.pt` plus mAP / precision / recall / confusion matrix.
2. Copy `best.pt` into `models/`, then `pip install -r requirements-yolo.txt` and restart.
   Startup log must say `Loaded YOLO weights`; `/health` must show `"detector_mode":"yolo"`.
3. The model's class names are mapped automatically (aliases in `app/constants.py`).

## API summary
| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | Main triage: `image_url`/`image_base64` + GPS/time/context -> status, issue, severity, priority, ETA, materials, explanation |
| POST | `/analyze/upload` | Same, multipart file upload |
| POST | `/verify-repair` | After-photo check; `verified:true` => close the ticket |
| POST | `/cluster` | DBSCAN duplicate merging into master incidents |
| POST | `/hotspots` | Hotspot grid + 7-day trend forecast |
| POST | `/complaint/classify` | Tamil / Hindi / English complaint -> category + urgency |
| POST | `/trust-score` | Reporter trust score and tier |
| POST | `/eta` | Standalone repair-time estimate |
| GET | `/health`, `/model/info`, `/demo`, `/docs` | Ops, metrics, demo UI, Swagger |

`status` values: `AUTO_DISPATCH`, `NEEDS_REVIEW` (low AI confidence -> human), `NEEDS_SECOND_REPORT` (low-trust reporter), `REJECTED`.

## Deploy
Docker (Hugging Face Spaces / Render / Railway): `docker build -t civiclens-ml . && docker run -p 7860:7860 civiclens-ml`.
Set `ML_API_KEY`, `CORS_ORIGINS`, and ideally `ALLOWED_IMAGE_HOSTS` (your image storage host). Give your friend
`ML_SERVICE_URL` + `ML_API_KEY`; he uses `integration/node_client.js` and `integration/ai_triage.sql`.
Free hosts sleep when idle: the first request can take ~30 s, so warm it up before a demo.

## Requirements checklist
| Requirement | Where |
|---|---|
| Fake / old / wrong-city photo detection | `exif_validator.py`, perceptual hash re-use check |
| Object detection + classification | `detector.py`, `training/` |
| Severity 1-5 (explainable) | `severity.py` |
| Resolution plan: materials, tools, crew | `materials.py` |
| Duration prediction | `eta_model.py` (simulated data, disclosed) |
| Duplicate clustering (DBSCAN) | `clustering.py` |
| After-photo closing loop | `/verify-repair` |
| Trust score | `trust.py`, `/trust-score` |
| Priority by location context | `priority.py` |
| Multilingual complaints | `complaint_nlp.py` |
| Hotspots + forecast | `hotspot.py` |
| Privacy (face/plate blur) | `privacy.py` |
| Human in the loop | `NEEDS_REVIEW` status |
| Security | API key, SSRF guard, size limits, CORS |
| Tests, Docker, docs, model card | `tests/`, `Dockerfile`, `README.md`, `MODEL_CARD.md` |

## Known limitations (state these in your report)
* ETA model is trained on **simulated** data (no public repair-time dataset exists).
* Severity is rule-based (area + class + count), not a trained regressor; no public dataset has 1-5 labels.
* Complaint NLP is a keyword baseline, not a fine-tuned language model.
* Face/plate blur uses Haar cascades: lightweight, will miss some faces and plates.
* No Grad-CAM; explainability is via the reasons list and the annotated boxes.
* Re-use detection keeps hashes in memory (resets on restart); store `image_hash` in the DB for production.
* Detection quality depends on the data you train with; night, rain and unusual angles will lower accuracy.
