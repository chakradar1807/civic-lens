"""Train the repair-time model on simulated data and print metrics for the report.

    python training/train_eta_model.py --rows 5000
To use REAL history instead, pass a CSV with columns:
    issue_type,severity,distance_to_depot_km,traffic_level,rain_mm,fix_minutes
    python training/train_eta_model.py --csv data/repairs.csv
"""
import argparse
import csv
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.eta_model import class_index, generate_synthetic_dataset, train  # noqa: E402


def load_csv(path):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            X.append([class_index(row["issue_type"].strip().lower()), float(row["severity"]),
                      float(row["distance_to_depot_km"]), float(row["traffic_level"]), float(row["rain_mm"])])
            y.append(float(row["fix_minutes"]))
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--csv", default=None)
    p.add_argument("--out-dir", default="models")
    a = p.parse_args()

    X, y = load_csv(a.csv) if a.csv else generate_synthetic_dataset(a.rows)
    model, metrics = train(X, y)
    if a.csv:
        metrics["data_source"] = f"real data from {a.csv}"
    os.makedirs(a.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(a.out_dir, "eta_model.joblib"))
    with open(os.path.join(a.out_dir, "eta_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
