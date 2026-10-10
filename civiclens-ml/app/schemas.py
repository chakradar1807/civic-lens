"""Request / response models (Pydantic v2). These define the API contract shared with the backend."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="Public http(s) URL of the uploaded photo")
    image_base64: Optional[str] = Field(None, description="Base64 (or data URI) of the photo")
    user_lat: Optional[float] = Field(None, ge=-90, le=90)
    user_lng: Optional[float] = Field(None, ge=-180, le=180)
    submitted_at: Optional[str] = Field(None, description="ISO-8601 timestamp; defaults to now")
    distance_to_depot_km: float = Field(5.0, ge=0, le=500)
    traffic_level: int = Field(1, ge=0, le=2, description="0 light, 1 normal, 2 heavy")
    rain_mm: float = Field(0.0, ge=0, le=500)
    near_sensitive: List[str] = Field(default_factory=list, description="e.g. ['hospital','school','main_road']")
    user_trust: Optional[float] = Field(None, ge=0, le=1, description="Reporter trust score from /trust-score")
    return_annotated: bool = True
    mock_issue_type: Optional[str] = Field(None, description="Only used in mock mode (demo/testing); 'none' simulates no detection")

    @model_validator(mode="after")
    def _need_image(self):
        if not self.image_url and not self.image_base64:
            raise ValueError("Provide image_url or image_base64")
        return self


class BBoxDetection(BaseModel):
    issue_type: str
    confidence: float
    bbox: List[int]


class ValidationInfo(BaseModel):
    exif_present: bool = False
    photo_age_hours: Optional[float] = None
    gps_distance_m: Optional[float] = None
    flags: List[str] = Field(default_factory=list)
    trust_penalty: float = 0.0


class AnalyzeResponse(BaseModel):
    valid: bool
    status: str  # AUTO_DISPATCH | NEEDS_REVIEW | NEEDS_SECOND_REPORT | REJECTED
    reject_reason: Optional[str] = None
    validation: ValidationInfo
    issue_type: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[int] = None
    severity_label: Optional[str] = None
    bbox: Optional[List[int]] = None
    image_size: Optional[List[int]] = None
    detections: List[BBoxDetection] = Field(default_factory=list)
    department: Optional[str] = None
    priority_score: Optional[int] = None
    priority_level: Optional[str] = None
    dispatch_policy: Optional[str] = None
    estimated_fix_minutes: Optional[int] = None
    estimated_fix_text: Optional[str] = None
    materials: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    crew_size: Optional[int] = None
    explanation: List[str] = Field(default_factory=list)
    privacy_regions_blurred: int = 0
    annotated_image_base64: Optional[str] = None
    image_hash: Optional[str] = None
    model_mode: str
    processing_ms: int


class VerifyRepairRequest(BaseModel):
    after_image_url: Optional[str] = None
    after_image_base64: Optional[str] = None
    issue_type: str = Field(..., description="The issue type of the original ticket")
    mock_issue_type: Optional[str] = Field(None, description="Mock mode only; 'none' simulates a fixed site")

    @model_validator(mode="after")
    def _need_image(self):
        if not self.after_image_url and not self.after_image_base64:
            raise ValueError("Provide after_image_url or after_image_base64")
        return self


class ReportPoint(BaseModel):
    id: str
    issue_type: str
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    timestamp: Optional[str] = None


class ClusterRequest(BaseModel):
    reports: List[ReportPoint] = Field(..., min_length=1, max_length=20000)
    radius_m: float = Field(50.0, gt=0, le=5000)
    window_minutes: float = Field(60.0, gt=0, le=100000)


class HotspotPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    timestamp: Optional[str] = None
    severity: Optional[int] = Field(None, ge=1, le=5)
    issue_type: Optional[str] = None


class HotspotRequest(BaseModel):
    reports: List[HotspotPoint] = Field(..., min_length=1, max_length=50000)
    cell_size_m: float = Field(500.0, ge=50, le=10000)
    top_n: int = Field(10, ge=1, le=100)


class ComplaintRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class TrustRequest(BaseModel):
    verified_id: bool = False
    total_reports: int = Field(0, ge=0)
    accepted_reports: int = Field(0, ge=0)
    rejected_fake_reports: int = Field(0, ge=0)


class EtaRequest(BaseModel):
    issue_type: str
    severity: int = Field(..., ge=1, le=5)
    distance_to_depot_km: float = Field(5.0, ge=0, le=500)
    traffic_level: int = Field(1, ge=0, le=2)
    rain_mm: float = Field(0.0, ge=0, le=500)


class Generic(BaseModel):
    data: Dict[str, Any] = {}
