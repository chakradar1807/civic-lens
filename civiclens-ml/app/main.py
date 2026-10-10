"""CivicLens ML service - FastAPI entry point.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs      Demo UI: http://localhost:8000/demo
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app import pipeline
from app.config import settings
from app.constants import CLASSES, DEPARTMENTS
from app.schemas import (AnalyzeRequest, AnalyzeResponse, ClusterRequest, ComplaintRequest, EtaRequest,
                         HotspotRequest, TrustRequest, VerifyRepairRequest)
from app.services import image_io
from app.services.clustering import cluster_reports
from app.services.complaint_nlp import classify_complaint
from app.services.detector import build_detector
from app.services.eta_model import EtaModel, format_minutes
from app.services.hotspot import build_hotspots
from app.services.trust import compute_trust

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("civiclens")

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are loaded ONCE at startup, never per request.
    STATE["detector"] = build_detector()
    STATE["eta"] = EtaModel().load_or_train()
    log.info("CivicLens ML ready (detector mode: %s)", STATE["detector"].mode)
    yield
    STATE.clear()


app = FastAPI(
    title="CivicLens ML Service",
    version="1.0.0",
    description="AI triage for citizen-reported civic issues: photo verification, detection, severity, "
                "priority, repair-time prediction, duplicate clustering, hotspots and multilingual intake.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


def require_key(x_api_key: Optional[str] = Header(None)):
    """If ML_API_KEY is set, every protected request must send it as the X-API-Key header."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------- meta
@app.get("/")
def root():
    return {"service": "CivicLens ML", "docs": "/docs", "demo": "/demo", "health": "/health"}


@app.get("/health")
def health():
    return {"ok": True, "detector_mode": STATE["detector"].mode, "eta_model_loaded": STATE["eta"].model is not None}


@app.get("/model/info")
def model_info():
    return {
        "classes": CLASSES,
        "departments": DEPARTMENTS,
        "detector_mode": STATE["detector"].mode,
        "yolo_weights": settings.yolo_weights,
        "thresholds": {"conf": settings.conf_threshold, "human_review_below": settings.review_threshold},
        "photo_rules": {"max_age_hours": settings.max_photo_age_hours, "max_gps_distance_m": settings.max_gps_distance_m},
        "eta_model_metrics": STATE["eta"].metrics,
    }


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    return (Path(__file__).parent / "static" / "demo.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- core triage
@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_key)])
def analyze_json(req: AnalyzeRequest):
    """Main endpoint used by the Node backend: send image_url (or base64) + metadata."""
    try:
        raw = pipeline.get_image_bytes(req.image_url, req.image_base64)
        return pipeline.analyze(raw, req, STATE["detector"], STATE["eta"])
    except image_io.ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/analyze/upload", response_model=AnalyzeResponse, dependencies=[Depends(require_key)])
def analyze_upload(
    file: UploadFile = File(...),
    user_lat: Optional[float] = Form(None),
    user_lng: Optional[float] = Form(None),
    submitted_at: Optional[str] = Form(None),
    distance_to_depot_km: float = Form(5.0),
    traffic_level: int = Form(1),
    rain_mm: float = Form(0.0),
    near_sensitive: Optional[str] = Form(None, description="comma separated, e.g. hospital,school"),
    user_trust: Optional[float] = Form(None),
    mock_issue_type: Optional[str] = Form(None),
):
    """Same as /analyze but accepts a multipart file upload (used by the demo page)."""
    raw = file.file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image is too large")
    try:
        req = AnalyzeRequest(
            image_base64="upload",  # placeholder: bytes come from the file itself
            user_lat=user_lat, user_lng=user_lng, submitted_at=submitted_at,
            distance_to_depot_km=distance_to_depot_km, traffic_level=traffic_level, rain_mm=rain_mm,
            near_sensitive=[s.strip() for s in (near_sensitive or "").split(",") if s.strip()],
            user_trust=user_trust, mock_issue_type=mock_issue_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        return pipeline.analyze(raw, req, STATE["detector"], STATE["eta"])
    except image_io.ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/verify-repair", dependencies=[Depends(require_key)])
def verify_repair(req: VerifyRepairRequest):
    """AI closing loop: confirm from the crew's after-photo that the issue is really gone."""
    try:
        raw = pipeline.get_image_bytes(req.after_image_url, req.after_image_base64)
        return pipeline.verify_repair(raw, req.issue_type, STATE["detector"], req.mock_issue_type)
    except image_io.ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------- analytics / intelligence
@app.post("/cluster", dependencies=[Depends(require_key)])
def cluster(req: ClusterRequest):
    """Merge duplicate reports (same type, within radius_m and window_minutes) into master incidents."""
    return cluster_reports([r.model_dump() for r in req.reports], req.radius_m, req.window_minutes)


@app.post("/hotspots", dependencies=[Depends(require_key)])
def hotspots(req: HotspotRequest):
    return build_hotspots([r.model_dump() for r in req.reports], req.cell_size_m, req.top_n)


@app.post("/complaint/classify", dependencies=[Depends(require_key)])
def complaint(req: ComplaintRequest):
    """Classify a free-text complaint (English / Tamil / Hindi) into issue type + urgency."""
    return classify_complaint(req.text)


@app.post("/trust-score", dependencies=[Depends(require_key)])
def trust(req: TrustRequest):
    return compute_trust(req.verified_id, req.total_reports, req.accepted_reports, req.rejected_fake_reports)


@app.post("/eta", dependencies=[Depends(require_key)])
def eta(req: EtaRequest):
    minutes = STATE["eta"].predict_minutes(
        req.issue_type.strip().lower(), req.severity, req.distance_to_depot_km, req.traffic_level, req.rain_mm
    )
    return {"estimated_fix_minutes": minutes, "estimated_fix_text": format_minutes(minutes)}
