"""Privacy protection: blur faces and number plates before any image is returned or stored.

Uses OpenCV's bundled Haar cascades (no downloads). Haar cascades are a lightweight baseline
and will miss some faces/plates - the README lists this as a limitation.
"""
import logging
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger("civiclens.privacy")
_cascades = {}


def _get_cascade(filename: str):
    if filename not in _cascades:
        path = cv2.data.haarcascades + filename
        cascade = cv2.CascadeClassifier(path)
        _cascades[filename] = None if cascade.empty() else cascade
    return _cascades[filename]


def blur_sensitive_regions(rgb: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (blurred copy, number of regions blurred)."""
    out = rgb.copy()
    count = 0
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for filename, min_neighbors in (
            ("haarcascade_frontalface_default.xml", 6),
            ("haarcascade_russian_plate_number.xml", 5),
        ):
            cascade = _get_cascade(filename)
            if cascade is None:
                continue
            boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(30, 30))
            for (x, y, w, h) in boxes:
                k = max(15, (min(w, h) // 2) | 1)
                roi = out[y:y + h, x:x + w]
                out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
                count += 1
    except Exception as exc:  # privacy step must never crash the pipeline
        log.warning("Blur step failed: %s", exc)
    return out, count
