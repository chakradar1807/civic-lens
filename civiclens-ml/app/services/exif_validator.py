"""Module A - photo authenticity checks using EXIF metadata.

Rules (deliberately lenient where phones legitimately strip metadata):
  * No EXIF at all          -> allowed, but flagged and trust is reduced
  * Photo older than limit  -> REJECT  (stale / reused internet photo)
  * Photo GPS far from the user's phone GPS -> REJECT (taken somewhere else)
  * Edited with photo-editing software -> flagged, trust reduced
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from PIL import Image

from app.config import settings
from app.services.geo import haversine_m
from app.services.timeutil import now_utc, parse_ts

TAG_EXIF_IFD = 0x8769
TAG_GPS_IFD = 0x8825
TAG_DATETIME = 306
TAG_SOFTWARE = 305
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME_DIGITIZED = 36868

EDITORS = ("photoshop", "gimp", "canva", "picsart", "snapseed", "lightroom", "pixlr", "facetune")


@dataclass
class ExifInfo:
    present: bool
    photo_time: Optional[datetime]  # naive, as written by the camera
    gps: Optional[Tuple[float, float]]
    software: Optional[str]


def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if str(ref).upper().startswith(("S", "W")):
        deg = -deg
    return deg


def _parse_exif_datetime(text) -> Optional[datetime]:
    if not text:
        return None
    if isinstance(text, bytes):
        text = text.decode(errors="ignore")
    text = str(text).replace("\x00", "").strip()
    try:
        return datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def read_exif(img: Image.Image) -> ExifInfo:
    try:
        exif = img.getexif()
    except Exception:
        return ExifInfo(False, None, None, None)
    if not exif:
        return ExifInfo(False, None, None, None)

    try:
        exif_ifd = exif.get_ifd(TAG_EXIF_IFD) or {}
    except Exception:
        exif_ifd = {}
    try:
        gps_ifd = exif.get_ifd(TAG_GPS_IFD) or {}
    except Exception:
        gps_ifd = {}

    photo_time = (
        _parse_exif_datetime(exif_ifd.get(TAG_DATETIME_ORIGINAL))
        or _parse_exif_datetime(exif_ifd.get(TAG_DATETIME_DIGITIZED))
        or _parse_exif_datetime(exif.get(TAG_DATETIME))
    )

    gps = None
    if gps_ifd.get(2) and gps_ifd.get(4):
        lat = _dms_to_deg(gps_ifd.get(2), gps_ifd.get(1, "N"))
        lng = _dms_to_deg(gps_ifd.get(4), gps_ifd.get(3, "E"))
        if lat is not None and lng is not None:
            gps = (lat, lng)

    software = exif.get(TAG_SOFTWARE)
    if isinstance(software, bytes):
        software = software.decode(errors="ignore")
    software = str(software).strip() if software else None

    return ExifInfo(True, photo_time, gps, software)


def validate_photo(
    info: ExifInfo,
    submitted_at: Optional[datetime],
    user_lat: Optional[float],
    user_lng: Optional[float],
) -> dict:
    """Return a dict: reject(bool), reason, flags, trust_penalty, photo_age_hours, gps_distance_m."""
    flags = []
    penalty = 0.0
    reject = False
    reason = None
    age_hours = None
    gps_distance = None
    submitted = submitted_at or now_utc()

    if not info.present or (info.photo_time is None and info.gps is None):
        flags.append("no_exif_metadata")
        penalty += 0.10
    else:
        if info.photo_time is not None:
            photo_utc = (info.photo_time - timedelta(hours=settings.exif_tz_offset_hours)).replace(
                tzinfo=timezone.utc
            )
            age_hours = round((submitted - photo_utc).total_seconds() / 3600.0, 2)
            if age_hours > settings.max_photo_age_hours:
                reject, reason = True, "stale_photo"
                flags.append("photo_older_than_allowed")
                penalty += 0.30
            elif age_hours < -2:
                flags.append("photo_timestamp_in_future")
                penalty += 0.10
        else:
            flags.append("no_photo_timestamp")
            penalty += 0.05

        if info.gps is not None and user_lat is not None and user_lng is not None:
            gps_distance = round(haversine_m(info.gps[0], info.gps[1], user_lat, user_lng), 1)
            if gps_distance > settings.max_gps_distance_m and not reject:
                reject, reason = True, "photo_location_mismatch"
                flags.append("photo_gps_far_from_user")
                penalty += 0.30
        elif info.gps is None:
            flags.append("no_photo_gps")

    if info.software and any(e in info.software.lower() for e in EDITORS):
        flags.append("edited_with_photo_software")
        penalty += 0.15

    return {
        "reject": reject,
        "reason": reason,
        "flags": flags,
        "trust_penalty": round(penalty, 2),
        "exif_present": bool(info.present),
        "photo_age_hours": age_hours,
        "gps_distance_m": gps_distance,
    }
