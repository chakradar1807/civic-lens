"""Module D - duplicate-report clustering with DBSCAN (haversine metric).

Reports of the same issue type within `radius_m` metres AND within `window_minutes` of each
other are merged into one master incident, so 50 citizens reporting the same fallen tree
create one ticket instead of 50.
"""
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from app.services.geo import EARTH_RADIUS_M
from app.services.timeutil import parse_ts

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def cluster_reports(reports: List[Dict], radius_m: float = 50.0, window_minutes: float = 60.0) -> Dict:
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in reports:
        by_type[r["issue_type"]].append(r)

    incidents = []
    counter = 0
    for issue_type, items in by_type.items():
        coords = np.radians(np.array([[r["lat"], r["lng"]] for r in items], dtype=float))
        labels = DBSCAN(
            eps=radius_m / EARTH_RADIUS_M, min_samples=1, metric="haversine", algorithm="ball_tree"
        ).fit_predict(coords)

        spatial: Dict[int, List[Dict]] = defaultdict(list)
        for label, r in zip(labels, items):
            spatial[int(label)].append(r)

        for group in spatial.values():
            group = sorted(group, key=lambda r: parse_ts(r.get("timestamp")) or _EPOCH)
            buckets: List[List[Dict]] = []
            for r in group:
                t = parse_ts(r.get("timestamp"))
                if buckets:
                    last_t = parse_ts(buckets[-1][-1].get("timestamp"))
                    if t is not None and last_t is not None and (t - last_t).total_seconds() / 60.0 > window_minutes:
                        buckets.append([r])
                        continue
                    buckets[-1].append(r)
                else:
                    buckets.append([r])

            for bucket in buckets:
                counter += 1
                times = [parse_ts(x.get("timestamp")) for x in bucket if parse_ts(x.get("timestamp"))]
                incidents.append({
                    "incident_id": f"INC-{counter:04d}",
                    "issue_type": issue_type,
                    "master_report_id": bucket[0]["id"],
                    "report_ids": [x["id"] for x in bucket],
                    "report_count": len(bucket),
                    "duplicates_merged": len(bucket) - 1,
                    "centroid": {
                        "lat": round(float(np.mean([x["lat"] for x in bucket])), 6),
                        "lng": round(float(np.mean([x["lng"] for x in bucket])), 6),
                    },
                    "first_seen": min(times).isoformat() if times else None,
                    "last_seen": max(times).isoformat() if times else None,
                })

    total = len(reports)
    n_inc = len(incidents)
    reduction = round((1 - n_inc / total) * 100, 1) if total else 0.0
    return {
        "total_reports": total,
        "incidents_created": n_inc,
        "ticket_reduction_percent": reduction,
        "radius_m": radius_m,
        "window_minutes": window_minutes,
        "incidents": incidents,
    }
