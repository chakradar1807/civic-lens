"""Module B - civic issue detection.

YoloDetector : real YOLOv8 model (needs models/best.pt and the `ultralytics` package)
MockDetector : deterministic stand-in so the whole pipeline and the frontend integration
               can be built and demoed before the real model is trained. Every response
               reports `model_mode`, so a mock result is never mistaken for a real one.
"""
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.config import settings
from app.constants import CLASSES, normalize_class

log = logging.getLogger("civiclens.detector")


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels of the processed image

    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


class BaseDetector:
    mode = "base"

    def detect(self, rgb: np.ndarray, mock_hint: Optional[str] = None) -> List[Detection]:
        raise NotImplementedError


class MockDetector(BaseDetector):
    mode = "mock"

    def detect(self, rgb: np.ndarray, mock_hint: Optional[str] = None) -> List[Detection]:
        if mock_hint and mock_hint.lower() == "none":
            return []
        h, w = rgb.shape[:2]
        digest = hashlib.md5(rgb[::16, ::16].tobytes()).digest()
        hint = normalize_class(mock_hint) if mock_hint else None
        cls = hint if hint in CLASSES else CLASSES[digest[0] % len(CLASSES)]
        conf = round(0.60 + (digest[1] / 255.0) * 0.35, 3)
        bw = int(w * (0.20 + (digest[2] / 255.0) * 0.30))
        bh = int(h * (0.15 + (digest[3] / 255.0) * 0.30))
        x1, y1 = max(0, (w - bw) // 2), max(0, (h - bh) // 2)
        return [Detection(cls, conf, (x1, y1, min(w, x1 + bw), min(h, y1 + bh)))]


class YoloDetector(BaseDetector):
    mode = "yolo"

    def __init__(self, weights: str):
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.model = YOLO(weights)
        self.names = self.model.names
        log.info("Loaded YOLO weights %s with classes %s", weights, self.names)

    def detect(self, rgb: np.ndarray, mock_hint: Optional[str] = None) -> List[Detection]:
        results = self.model.predict(rgb, conf=settings.conf_threshold, verbose=False)
        out: List[Detection] = []
        if not results:
            return out
        res = results[0]
        h, w = rgb.shape[:2]
        for box in res.boxes:
            cls_id = int(box.cls[0])
            name = normalize_class(res.names.get(cls_id, str(cls_id)) if isinstance(res.names, dict)
                                   else res.names[cls_id])
            x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0].tolist())
            out.append(Detection(name, round(float(box.conf[0]), 3),
                                 (max(0, x1), max(0, y1), min(w, x2), min(h, y2))))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out


def build_detector() -> BaseDetector:
    mode = settings.detector_mode
    weights_exist = os.path.isfile(settings.yolo_weights)
    if mode == "mock":
        log.warning("DETECTOR_MODE=mock - using the mock detector (demo only).")
        return MockDetector()
    if mode in ("yolo", "auto") and weights_exist:
        try:
            return YoloDetector(settings.yolo_weights)
        except Exception as exc:  # missing package, corrupt weights...
            if mode == "yolo":
                raise
            log.warning("Could not load YOLO (%s). Falling back to mock detector.", exc)
            return MockDetector()
    if mode == "yolo":
        raise FileNotFoundError(f"YOLO weights not found at {settings.yolo_weights}")
    log.warning("No weights at %s - using the mock detector (demo only).", settings.yolo_weights)
    return MockDetector()
