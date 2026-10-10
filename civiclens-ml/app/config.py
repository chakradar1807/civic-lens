"""Central configuration. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _list(name: str) -> list:
    raw = os.getenv(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Settings:
    # Detector: "auto" (YOLO if weights exist, else mock), "yolo" (fail if missing), "mock"
    detector_mode: str = os.getenv("DETECTOR_MODE", "auto").lower()
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", str(BASE_DIR / "models" / "best.pt"))
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "0.50"))

    # Image handling
    max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", "1280"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    url_timeout_s: float = float(os.getenv("URL_TIMEOUT_S", "10"))
    allowed_image_hosts: list = field(default_factory=lambda: _list("ALLOWED_IMAGE_HOSTS"))
    allow_private_urls: bool = _bool("ALLOW_PRIVATE_URLS", False)

    # Photo authenticity rules
    max_photo_age_hours: float = float(os.getenv("MAX_PHOTO_AGE_HOURS", "24"))
    max_gps_distance_m: float = float(os.getenv("MAX_GPS_DISTANCE_M", "500"))
    exif_tz_offset_hours: float = float(os.getenv("EXIF_TZ_OFFSET_HOURS", "5.5"))  # IST
    reject_reused_images: bool = _bool("REJECT_REUSED_IMAGES", False)

    # Privacy
    blur_sensitive: bool = _bool("BLUR_SENSITIVE", True)

    # Models / paths
    models_dir: str = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))

    # Security / CORS
    api_key: str = os.getenv("ML_API_KEY", "")
    cors_origins: list = field(default_factory=lambda: _list("CORS_ORIGINS") or ["*"])


settings = Settings()
