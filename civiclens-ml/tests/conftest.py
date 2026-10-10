import io
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

# Force deterministic test settings BEFORE the app is imported.
os.environ["DETECTOR_MODE"] = "mock"
os.environ["MODELS_DIR"] = tempfile.mkdtemp(prefix="civiclens_models_")
os.environ["ML_API_KEY"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def make_jpeg(seed=0, taken=None, gps=None, software=None, size=(640, 480)) -> bytes:
    """Create a JPEG with optional EXIF: taken=(naive local datetime), gps=(lat, lng)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    arr[100:300, 150:450] = (90, 80, 70)  # a flat block so images differ structurally by seed
    img = Image.fromarray(arr)
    exif = Image.Exif()
    if taken is not None:
        exif_ifd = exif.get_ifd(0x8769)
        exif_ifd[36867] = taken.strftime("%Y:%m:%d %H:%M:%S")
    if software:
        exif[305] = software
    if gps is not None:
        lat, lng = gps
        gps_ifd = exif.get_ifd(0x8825)

        def dms(v):
            v = abs(v)
            d = int(v)
            m = int((v - d) * 60)
            s = round((v - d - m / 60) * 3600, 4)
            return (float(d), float(m), float(s))

        gps_ifd[1] = "N" if lat >= 0 else "S"
        gps_ifd[2] = dms(lat)
        gps_ifd[3] = "E" if lng >= 0 else "W"
        gps_ifd[4] = dms(lng)
    buf = io.BytesIO()
    if taken is not None or gps is not None or software:
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


def ist_now_naive() -> datetime:
    """Current time in IST as a naive datetime (what a phone in Chennai would write in EXIF)."""
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=5.5)
