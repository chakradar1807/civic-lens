"""Module C - rule-based severity (1-5) with human-readable reasons.

Severity = base severity of the issue type
         + size bonus (how much of the photo the defect covers)
         + spread bonus (several separate instances in one photo)
The reasons list is returned to the citizen/officer, which makes the decision explainable.
"""
from typing import List, Tuple

from app.constants import BASE_SEVERITY, SEVERITY_LABELS


def compute_severity(cls: str, bbox: tuple, image_size: Tuple[int, int], n_instances: int = 1):
    w, h = image_size
    img_area = max(1, w * h)
    box_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    ratio = box_area / img_area

    reasons: List[str] = []
    base = BASE_SEVERITY.get(cls, 2)
    reasons.append(f"Base severity for '{cls}' is {base}")

    if ratio >= 0.15:
        bonus = 2
        reasons.append(f"Large affected area ({ratio:.0%} of the photo): +2")
    elif ratio >= 0.05:
        bonus = 1
        reasons.append(f"Medium affected area ({ratio:.0%} of the photo): +1")
    else:
        bonus = 0
        reasons.append(f"Small affected area ({ratio:.0%} of the photo): +0")

    spread = 0
    if n_instances >= 3:
        spread = 1
        reasons.append(f"{n_instances} separate instances detected: +1")

    severity = max(1, min(5, base + bonus + spread))
    return severity, SEVERITY_LABELS[severity], round(ratio, 4), reasons
