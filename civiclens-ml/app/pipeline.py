"""Orchestrates the full triage pipeline:
   load image -> EXIF validation -> privacy blur -> detection -> severity -> priority
   -> ETA -> materials -> ticket status decision -> annotated preview.
"""
import time
from collections import OrderedDict
from typing import Optional

from app.config import settings
from app.constants import CLASSES, normalize_class
from app.schemas import AnalyzeRequest
from app.services import image_io
from app.services.annotate import draw_detections
from app.services.detector import BaseDetector
from app.services.eta_model import EtaModel, format_minutes
from app.services.exif_validator import read_exif, validate_photo
from app.services.materials import get_resolution_plan
from app.services.priority import compute_priority
from app.services.privacy import blur_sensitive_regions
from app.services.severity import compute_severity
from app.services.timeutil import parse_ts


class RecentHashes:
    """In-memory registry of recent image hashes to spot re-used photos (reset on restart).
    For production keep these hashes in the database (e.g. a BIGINT column + index)."""

    def __init__(self, max_items: int = 5000):
        self.items: "OrderedDict[int, bool]" = OrderedDict()
        self.max_items = max_items

    def seen_similar(self, h: int, max_distance: int = 4) -> bool:
        return any(image_io.hamming(h, other) <= max_distance for other in self.items)

    def add(self, h: int) -> None:
        self.items[h] = True
        self.items.move_to_end(h)
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)


recent_hashes = RecentHashes()


def get_image_bytes(image_url: Optional[str], image_base64: Optional[str]) -> bytes:
    if image_url:
        return image_io.load_bytes_from_url(image_url)
    if image_base64:
        return image_io.load_bytes_from_base64(image_base64)
    raise image_io.ImageError("No image supplied")


def analyze(raw: bytes, req: AnalyzeRequest, detector: BaseDetector, eta: EtaModel) -> dict:
    started = time.perf_counter()
    pil = image_io.open_image(raw)

    # --- Module A: authenticity (uses the ORIGINAL file's EXIF) ---
    exif_info = read_exif(pil)
    validation = validate_photo(exif_info, parse_ts(req.submitted_at), req.user_lat, req.user_lng)
    img_hash = image_io.dhash(pil)
    reused = recent_hashes.seen_similar(img_hash)
    if reused:
        validation["flags"].append("possible_reused_image")
        validation["trust_penalty"] = round(validation["trust_penalty"] + 0.20, 2)
        if settings.reject_reused_images and not validation["reject"]:
            validation["reject"], validation["reason"] = True, "reused_image"
    recent_hashes.add(img_hash)

    rgb = image_io.prepare_rgb(pil)
    size = image_io.image_size(rgb)
    hash_hex = f"{img_hash:016x}"
    val_out = {
        "exif_present": validation["exif_present"],
        "photo_age_hours": validation["photo_age_hours"],
        "gps_distance_m": validation["gps_distance_m"],
        "flags": validation["flags"],
        "trust_penalty": validation["trust_penalty"],
    }

    def finish(payload: dict) -> dict:
        payload.update({
            "validation": val_out,
            "image_hash": hash_hex,
            "image_size": list(size),
            "model_mode": detector.mode,
            "processing_ms": int((time.perf_counter() - started) * 1000),
        })
        return payload

    if validation["reject"]:
        return finish({
            "valid": False, "status": "REJECTED", "reject_reason": validation["reason"],
            "explanation": [f"Photo rejected: {validation['reason']}"],
        })

    # --- Privacy: blur faces / plates before anything is returned ---
    blurred_count = 0
    work = rgb
    if settings.blur_sensitive:
        work, blurred_count = blur_sensitive_regions(rgb)

    # --- Module B: detection ---
    detections = detector.detect(rgb, mock_hint=req.mock_issue_type)
    if not detections:
        return finish({
            "valid": False, "status": "REJECTED", "reject_reason": "no_civic_issue_detected",
            "explanation": ["No pothole, garbage, streetlight or waterlogging issue was found in the photo."],
            "privacy_regions_blurred": blurred_count,
        })

    primary = detections[0]
    same_class = [d for d in detections if d.cls == primary.cls]

    # --- Module C: severity, priority ---
    severity, sev_label, area_ratio, sev_reasons = compute_severity(primary.cls, primary.bbox, size, len(same_class))
    p_score, p_level, policy, p_reasons = compute_priority(severity, primary.conf, req.near_sensitive)

    # --- Module E: ETA + materials ---
    minutes = eta.predict_minutes(primary.cls, severity, req.distance_to_depot_km, req.traffic_level, req.rain_mm)
    plan = get_resolution_plan(primary.cls, severity)

    # --- Ticket status decision ---
    explanation = sev_reasons + p_reasons
    if primary.conf < settings.review_threshold:
        status = "NEEDS_REVIEW"
        explanation.append(f"AI confidence {primary.conf:.0%} is below {settings.review_threshold:.0%}: sent to a human reviewer")
    elif req.user_trust is not None and (req.user_trust - validation["trust_penalty"]) < 0.5:
        status = "NEEDS_SECOND_REPORT"
        explanation.append("Reporter trust is low: waiting for a second independent report or officer check")
    else:
        status = "AUTO_DISPATCH"
        explanation.append("Photo verified and AI confidence is sufficient: eligible for automatic dispatch")

    # --- Annotated preview (privacy-blurred image + boxes) ---
    annotated = None
    if req.return_annotated:
        annotated = image_io.encode_jpeg_base64(draw_detections(work, detections, severity))

    return finish({
        "valid": True,
        "status": status,
        "reject_reason": None,
        "issue_type": primary.cls,
        "confidence": primary.conf,
        "severity": severity,
        "severity_label": sev_label,
        "bbox": [int(v) for v in primary.bbox],
        "detections": [{"issue_type": d.cls, "confidence": d.conf, "bbox": [int(v) for v in d.bbox]} for d in detections],
        "department": plan["department"],
        "priority_score": p_score,
        "priority_level": p_level,
        "dispatch_policy": policy,
        "estimated_fix_minutes": minutes,
        "estimated_fix_text": format_minutes(minutes),
        "materials": plan["materials"],
        "tools": plan["tools"],
        "crew_size": plan["crew_size"],
        "explanation": explanation,
        "privacy_regions_blurred": blurred_count,
        "annotated_image_base64": annotated,
    })


def verify_repair(raw: bytes, issue_type: str, detector: BaseDetector, mock_hint: Optional[str] = None) -> dict:
    """Run the detector on the worker's 'after' photo. Verified = the original issue is no longer found."""
    started = time.perf_counter()
    pil = image_io.open_image(raw)
    rgb = image_io.prepare_rgb(pil)
    target = normalize_class(issue_type)
    detections = detector.detect(rgb, mock_hint=mock_hint)
    remaining = [d for d in detections if d.cls == target and d.conf >= settings.review_threshold]
    verified = len(remaining) == 0
    return {
        "verified": verified,
        "status": "RESOLVED" if verified else "REPAIR_NOT_VERIFIED",
        "issue_type": target,
        "remaining_detections": [{"issue_type": d.cls, "confidence": d.conf, "bbox": [int(v) for v in d.bbox]} for d in remaining],
        "message": "No remaining issue detected - ticket can be closed and the citizen notified."
        if verified else "The issue is still visible in the after-photo - ask the crew to retake or redo the repair.",
        "model_mode": detector.mode,
        "processing_ms": int((time.perf_counter() - started) * 1000),
    }
