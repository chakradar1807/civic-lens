# Model Card - CivicLens detector and triage models

## Intended use
Assist municipal staff by pre-sorting citizen photos of civic problems. Decisions that affect people
(e.g. throttling a user) must be reviewable by a human; low-confidence results go to `NEEDS_REVIEW`.
Not intended for surveillance, identifying individuals, or enforcement.

## Models
| Component | Type | Training data | Notes |
|---|---|---|---|
| Issue detector | YOLOv8n (Ultralytics) | Public datasets merged into 4 classes: pothole, garbage, damaged_streetlight, waterlogging (fill in dataset names, licenses, image counts) | Metrics from `training/train_yolo.py` |
| Severity | Rule-based | n/a | Base severity + affected area + instance count |
| Repair time | RandomForestRegressor | **Simulated** (formula + noise) | MAE / RMSE / R2 in `/model/info` |
| Duplicate clustering | DBSCAN (haversine) | n/a | radius 50 m, window 60 min (configurable) |
| Complaint NLP | Keyword baseline | n/a | English, Tamil, Hindi (script + romanised) |
| Trust score | Smoothed formula | n/a | Identity prior + acceptance rate - fake-report penalty |

## Evaluation (fill in after training)
mAP@0.5 = ___ , mAP@0.5:0.95 = ___ , precision = ___ , recall = ___ ; confusion matrix and sample predictions from `runs/civiclens/yolov8n/`.
Authenticity checks: report how many of N stale/other-city test photos were caught.

## Privacy
Faces and number plates are blurred (best effort) before annotated images are returned. Do not store
raw citizen photos longer than needed; GPS is used only for the location check.

## Limitations and risks
* Lower accuracy at night, in heavy rain, in unusual angles, or on road types missing from training data.
* Missing EXIF is allowed (many phones strip it) and only lowers trust, so spoofed metadata can pass.
* Class imbalance can make rarer classes (e.g. waterlogging) less reliable; check per-class metrics.
* Review ward-level outcomes regularly for unequal response times or detection quality across areas.
