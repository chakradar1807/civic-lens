"""Hotspot map + simple next-week forecast.

Reports are binned into grid cells (default ~500 m). Each cell gets a risk score from report
count and severity. A linear trend over daily report counts gives a naive 7-day forecast;
it is a transparent baseline, not a deep model - state that when presenting.
"""
import math
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List

import numpy as np

from app.services.timeutil import parse_ts


def build_hotspots(reports: List[Dict], cell_size_m: float = 500.0, top_n: int = 10) -> Dict:
    if not reports:
        return {"total_reports": 0, "cells": [], "forecast": None}

    lat0 = float(np.mean([r["lat"] for r in reports]))
    dlat = cell_size_m / 111_320.0
    dlng = cell_size_m / (111_320.0 * max(0.01, math.cos(math.radians(lat0))))

    cells: Dict[tuple, Dict] = defaultdict(lambda: {"count": 0, "severity_sum": 0.0, "lats": [], "lngs": [], "types": defaultdict(int)})
    for r in reports:
        key = (int(math.floor(r["lat"] / dlat)), int(math.floor(r["lng"] / dlng)))
        c = cells[key]
        c["count"] += 1
        c["severity_sum"] += float(r.get("severity") or 1)
        c["lats"].append(r["lat"])
        c["lngs"].append(r["lng"])
        c["types"][r.get("issue_type") or "unknown"] += 1

    max_count = max(c["count"] for c in cells.values())
    ranked = []
    for c in cells.values():
        avg_sev = c["severity_sum"] / c["count"]
        risk = round(100 * (0.6 * c["count"] / max_count + 0.4 * avg_sev / 5.0), 1)
        top_type = max(c["types"].items(), key=lambda kv: kv[1])[0]
        ranked.append({
            "center": {"lat": round(float(np.mean(c["lats"])), 6), "lng": round(float(np.mean(c["lngs"])), 6)},
            "report_count": c["count"],
            "avg_severity": round(avg_sev, 2),
            "top_issue_type": top_type,
            "risk_score": risk,
            "risk_level": "HIGH" if risk >= 70 else "MEDIUM" if risk >= 40 else "LOW",
        })
    ranked.sort(key=lambda x: x["risk_score"], reverse=True)

    return {
        "total_reports": len(reports),
        "cell_size_m": cell_size_m,
        "cells": ranked[:top_n],
        "forecast": _forecast(reports),
    }


def _forecast(reports: List[Dict]):
    times = [parse_ts(r.get("timestamp")) for r in reports]
    times = [t for t in times if t is not None]
    if not times:
        return None
    days = sorted({t.date() for t in times})
    if len(days) < 3:
        return {"available": False, "reason": "need reports spanning at least 3 different days"}

    start = days[0]
    span = (days[-1] - start).days + 1
    counts = np.zeros(span)
    for t in times:
        counts[(t.date() - start).days] += 1
    x = np.arange(span, dtype=float)
    slope, intercept = np.polyfit(x, counts, 1)
    future_x = np.arange(span, span + 7, dtype=float)
    pred = np.clip(slope * future_x + intercept, 0, None)
    trend = "rising" if slope > 0.1 else "falling" if slope < -0.1 else "stable"
    return {
        "available": True,
        "method": "linear trend on daily report counts (baseline)",
        "trend": trend,
        "slope_reports_per_day": round(float(slope), 3),
        "next_7_days": [
            {"date": (start + timedelta(days=int(fx))).isoformat(), "expected_reports": round(float(p), 1)}
            for fx, p in zip(future_x, pred)
        ],
    }
