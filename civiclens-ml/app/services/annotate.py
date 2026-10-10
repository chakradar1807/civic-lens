"""Draw detection boxes and labels on an image (RGB numpy array)."""
from typing import List

import cv2
import numpy as np

from app.services.detector import Detection

SEVERITY_COLORS = {1: (46, 160, 67), 2: (130, 180, 40), 3: (240, 180, 20), 4: (240, 110, 20), 5: (220, 30, 30)}


def draw_detections(rgb: np.ndarray, detections: List[Detection], primary_severity: int = 3) -> np.ndarray:
    out = rgb.copy()
    thickness = max(2, int(round(max(out.shape[:2]) / 400)))
    scale = max(0.5, max(out.shape[:2]) / 1200)
    color = SEVERITY_COLORS.get(primary_severity, (240, 180, 20))
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        c = color if i == 0 else (90, 140, 230)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        label = f"{det.cls} {det.conf:.0%}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + base + 4), c, -1)
        cv2.putText(out, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out
