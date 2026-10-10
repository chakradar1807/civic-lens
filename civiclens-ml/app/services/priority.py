"""Priority score (0-100): severity + context. A pothole next to a hospital outranks one in a dead-end lane."""
from typing import Iterable, List, Tuple

from app.constants import DISPATCH_POLICY

SENSITIVE_WEIGHTS = {
    "hospital": 15, "school": 12, "highway": 12, "main_road": 10, "college": 8,
    "junction": 6, "bus_stop": 5, "market": 5, "metro_station": 6, "residential": 2,
}


def compute_priority(severity: int, confidence: float, near_sensitive: Iterable[str]) -> Tuple[int, str, str, List[str]]:
    reasons: List[str] = []
    score = severity * 12  # 12..60
    reasons.append(f"Severity {severity} contributes {severity * 12} points")

    bonus = 0
    seen = set()
    for place in near_sensitive or []:
        key = str(place).strip().lower().replace(" ", "_")
        if key in SENSITIVE_WEIGHTS and key not in seen:
            seen.add(key)
            bonus += SENSITIVE_WEIGHTS[key]
            reasons.append(f"Near {key.replace('_', ' ')}: +{SENSITIVE_WEIGHTS[key]}")
    bonus = min(bonus, 30)
    score += bonus

    conf_points = int(round(max(0.0, min(1.0, confidence)) * 10))
    score += conf_points
    score = int(max(0, min(100, score)))

    if score >= 80:
        level = "CRITICAL"
    elif score >= 60:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level, DISPATCH_POLICY[level], reasons
