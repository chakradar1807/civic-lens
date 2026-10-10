import base64
from datetime import datetime, timedelta, timezone

from tests.conftest import ist_now_naive, make_jpeg

CHENNAI = (13.0827, 80.2707)


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def post_analyze(client, raw, **extra):
    body = {"image_base64": b64(raw), "mock_issue_type": "pothole"}
    body.update(extra)
    return client.post("/analyze", json=body)


# ------------------------------------------------------------------ meta
def test_health_and_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    info = client.get("/model/info").json()
    assert "pothole" in info["classes"]
    assert info["eta_model_metrics"]["r2"] > 0.8


def test_demo_page_served(client):
    r = client.get("/demo")
    assert r.status_code == 200 and "CivicLens" in r.text


# ------------------------------------------------------------------ Module A: authenticity
def test_fresh_photo_with_matching_gps_is_accepted(client):
    raw = make_jpeg(1, taken=ist_now_naive(), gps=CHENNAI)
    r = post_analyze(client, raw, user_lat=CHENNAI[0], user_lng=CHENNAI[1])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["valid"] is True and d["status"] == "AUTO_DISPATCH"
    assert d["issue_type"] == "pothole"
    assert d["validation"]["gps_distance_m"] is not None and d["validation"]["gps_distance_m"] < 50


def test_stale_photo_is_rejected(client):
    old = ist_now_naive() - timedelta(days=3 * 365)
    r = post_analyze(client, make_jpeg(2, taken=old, gps=CHENNAI))
    d = r.json()
    assert r.status_code == 200
    assert d["valid"] is False and d["reject_reason"] == "stale_photo" and d["status"] == "REJECTED"


def test_photo_from_another_city_is_rejected(client):
    mumbai = (19.0760, 72.8777)
    raw = make_jpeg(3, taken=ist_now_naive(), gps=mumbai)
    d = post_analyze(client, raw, user_lat=CHENNAI[0], user_lng=CHENNAI[1]).json()
    assert d["valid"] is False and d["reject_reason"] == "photo_location_mismatch"


def test_missing_exif_is_allowed_but_flagged(client):
    d = post_analyze(client, make_jpeg(4)).json()
    assert d["valid"] is True
    assert "no_exif_metadata" in d["validation"]["flags"]
    assert d["validation"]["trust_penalty"] > 0


def test_edited_photo_is_flagged(client):
    raw = make_jpeg(5, taken=ist_now_naive(), software="Adobe Photoshop 25")
    d = post_analyze(client, raw).json()
    assert "edited_with_photo_software" in d["validation"]["flags"]


def test_reused_image_is_flagged(client):
    raw = make_jpeg(6)
    post_analyze(client, raw)
    d = post_analyze(client, raw).json()
    assert "possible_reused_image" in d["validation"]["flags"]


# ------------------------------------------------------------------ Modules B, C, E: triage
def test_full_triage_fields(client):
    d = post_analyze(client, make_jpeg(7, taken=ist_now_naive()), near_sensitive=["hospital", "school"]).json()
    assert 1 <= d["severity"] <= 5
    assert len(d["bbox"]) == 4
    assert d["estimated_fix_minutes"] >= 10 and d["estimated_fix_text"]
    assert d["materials"] and d["tools"] and d["crew_size"] >= 1
    assert d["department"] == "Roads & Highways"
    assert d["annotated_image_base64"].startswith("data:image/jpeg;base64,")
    assert d["model_mode"] == "mock"
    assert any("hospital" in e for e in d["explanation"])


def test_no_issue_detected_is_rejected(client):
    d = post_analyze(client, make_jpeg(8), mock_issue_type="none").json()
    assert d["valid"] is False and d["reject_reason"] == "no_civic_issue_detected"


def test_low_trust_user_needs_second_report(client):
    d = post_analyze(client, make_jpeg(9, taken=ist_now_naive()), user_trust=0.4).json()
    assert d["status"] == "NEEDS_SECOND_REPORT"


def test_upload_endpoint(client):
    raw = make_jpeg(10, taken=ist_now_naive())
    r = client.post("/analyze/upload", files={"file": ("a.jpg", raw, "image/jpeg")},
                    data={"mock_issue_type": "garbage", "near_sensitive": "school"})
    assert r.status_code == 200, r.text
    assert r.json()["issue_type"] == "garbage"


def test_bad_inputs(client):
    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze", json={"image_base64": b64(b"not an image")}).status_code == 400
    assert client.post("/analyze", json={"image_url": "http://127.0.0.1/x.jpg"}).status_code == 400
    assert client.post("/analyze", json={"image_url": "ftp://example.com/x.jpg"}).status_code == 400


def test_severity_grows_with_bigger_defect():
    from app.services.severity import compute_severity
    small = compute_severity("pothole", (0, 0, 50, 50), (1000, 1000))[0]
    large = compute_severity("pothole", (0, 0, 600, 600), (1000, 1000))[0]
    assert large > small and 1 <= small <= 5 and 1 <= large <= 5


# ------------------------------------------------------------------ verify repair
def test_verify_repair(client):
    raw = b64(make_jpeg(11))
    fixed = client.post("/verify-repair", json={"after_image_base64": raw, "issue_type": "pothole", "mock_issue_type": "none"}).json()
    assert fixed["verified"] is True and fixed["status"] == "RESOLVED"
    broken = client.post("/verify-repair", json={"after_image_base64": raw, "issue_type": "pothole", "mock_issue_type": "pothole"}).json()
    assert broken["verified"] is False


# ------------------------------------------------------------------ Module D: clustering
def test_dbscan_merges_duplicates(client):
    t0 = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)
    reports = []
    for i in range(10):  # 10 reports of the same fallen pothole within ~20 m
        reports.append({"id": f"r{i}", "issue_type": "pothole", "lat": 13.0827 + i * 0.00001,
                        "lng": 80.2707, "timestamp": (t0 + timedelta(minutes=i)).isoformat()})
    reports.append({"id": "far", "issue_type": "pothole", "lat": 13.10, "lng": 80.30, "timestamp": t0.isoformat()})
    reports.append({"id": "other", "issue_type": "garbage", "lat": 13.0827, "lng": 80.2707, "timestamp": t0.isoformat()})
    reports.append({"id": "late", "issue_type": "pothole", "lat": 13.0827, "lng": 80.2707,
                    "timestamp": (t0 + timedelta(hours=5)).isoformat()})
    d = client.post("/cluster", json={"reports": reports, "radius_m": 50, "window_minutes": 60}).json()
    assert d["total_reports"] == 13
    assert d["incidents_created"] == 4  # main cluster, far, other type, late (outside time window)
    main = max(d["incidents"], key=lambda x: x["report_count"])
    assert main["report_count"] == 10 and main["duplicates_merged"] == 9 and main["master_report_id"] == "r0"
    assert d["ticket_reduction_percent"] > 20


# ------------------------------------------------------------------ hotspots, NLP, trust, ETA
def test_hotspots_and_forecast(client):
    base = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    reports = []
    for day in range(7):
        for k in range(day + 1):  # rising trend
            reports.append({"lat": 13.0827 + k * 0.0001, "lng": 80.2707, "severity": 3,
                            "issue_type": "pothole", "timestamp": (base + timedelta(days=day)).isoformat()})
    reports.append({"lat": 13.2, "lng": 80.4, "severity": 1, "issue_type": "garbage", "timestamp": base.isoformat()})
    d = client.post("/hotspots", json={"reports": reports, "cell_size_m": 500, "top_n": 5}).json()
    assert d["cells"][0]["report_count"] >= 20 and d["cells"][0]["top_issue_type"] == "pothole"
    assert d["forecast"]["available"] is True and d["forecast"]["trend"] == "rising"
    assert len(d["forecast"]["next_7_days"]) == 7


def test_complaint_multilingual(client):
    cases = {
        "There is a big pothole near the school, very dangerous": ("pothole", "en/romanised"),
        "road la periya kuzhi irukku, accident aagum": ("pothole", "en/romanised"),
        "தெரு விளக்கு எரியவில்லை, இருட்டு": ("damaged_streetlight", "ta"),
        "यहाँ बहुत कचरा है, गंदगी फैल रही है": ("garbage", "hi"),
        "heavy waterlogging, drain blocked": ("waterlogging", "en/romanised"),
    }
    for text, (cat, lang) in cases.items():
        d = client.post("/complaint/classify", json={"text": text}).json()
        assert d["category"] == cat, (text, d)
        assert d["language"] == lang, (text, d)
    assert client.post("/complaint/classify", json={"text": "hello there"}).json()["category"] == "unknown"
    assert client.post("/complaint/classify", json={"text": "pothole near school, danger"}).json()["urgency"] == "high"


def test_trust_score(client):
    verified = client.post("/trust-score", json={"verified_id": True, "total_reports": 20, "accepted_reports": 19}).json()
    guest = client.post("/trust-score", json={"verified_id": False}).json()
    faker = client.post("/trust-score", json={"verified_id": False, "total_reports": 10, "accepted_reports": 1, "rejected_fake_reports": 6}).json()
    assert verified["tier"] == "HIGH" and guest["tier"] == "LOW" and faker["throttle"] is True
    assert verified["trust_score"] > guest["trust_score"] > faker["trust_score"]


def test_eta_behaviour(client):
    quick = client.post("/eta", json={"issue_type": "pothole", "severity": 1, "distance_to_depot_km": 1}).json()
    slow = client.post("/eta", json={"issue_type": "pothole", "severity": 5, "distance_to_depot_km": 20, "traffic_level": 2}).json()
    assert slow["estimated_fix_minutes"] > quick["estimated_fix_minutes"]


def test_api_key_enforced(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "api_key", "secret")
    assert client.post("/trust-score", json={}).status_code == 401
    assert client.post("/trust-score", json={}, headers={"X-API-Key": "secret"}).status_code == 200
