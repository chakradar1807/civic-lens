"""Module E - repair-time prediction.

No public dataset of real repair times exists, so the model is trained on SIMULATED data
generated from a documented formula plus noise. This is stated in the API (`/model/info`),
the model card and the README. Swap in real historical tickets later by calling
`train(X, y)` with real data - nothing else changes.
"""
import json
import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import settings
from app.constants import CLASSES

log = logging.getLogger("civiclens.eta")

BASE_MINUTES = {"pothole": 60, "garbage": 45, "damaged_streetlight": 50, "waterlogging": 90}
FEATURES = ["issue_type_idx", "severity", "distance_to_depot_km", "traffic_level", "rain_mm"]


def class_index(cls: str) -> int:
    return CLASSES.index(cls) if cls in CLASSES else -1


def generate_synthetic_dataset(n: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Documented formula: base * (0.55 + 0.25*severity) + travel time + rain delay + noise."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(CLASSES), n)
    severity = rng.integers(1, 6, n)
    dist = rng.uniform(0.5, 25.0, n)
    traffic = rng.integers(0, 3, n)  # 0 light, 1 normal, 2 heavy
    rain = rng.exponential(2.0, n).clip(0, 40)
    y = np.empty(n)
    for i in range(n):
        base = BASE_MINUTES[CLASSES[idx[i]]]
        work = base * (0.55 + 0.25 * severity[i])
        travel = dist[i] * 2.5 * (1 + 0.4 * traffic[i])
        rain_factor = 2.0 if CLASSES[idx[i]] == "waterlogging" else 1.2
        y[i] = work + travel + rain[i] * rain_factor + rng.normal(0, 8)
    y = np.clip(y, 10, None)
    X = np.column_stack([idx, severity, dist, traffic, rain]).astype(float)
    return X, y


def train(X: np.ndarray, y: np.ndarray, seed: int = 42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "mae_minutes": round(float(mean_absolute_error(y_te, pred)), 2),
        "rmse_minutes": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
        "r2": round(float(r2_score(y_te, pred)), 4),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "data_source": "simulated (formula + noise); replace with real repair history when available",
        "model": "RandomForestRegressor(n_estimators=150)",
    }
    return model, metrics


def format_minutes(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    if h and m:
        return f"{h} h {m} min"
    if h:
        return f"{h} h"
    return f"{m} min"


class EtaModel:
    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.metrics: Dict = {}
        self.path = os.path.join(settings.models_dir, "eta_model.joblib")
        self.metrics_path = os.path.join(settings.models_dir, "eta_metrics.json")

    def load_or_train(self) -> "EtaModel":
        if os.path.isfile(self.path):
            try:
                self.model = joblib.load(self.path)
                if os.path.isfile(self.metrics_path):
                    with open(self.metrics_path, "r", encoding="utf-8") as fh:
                        self.metrics = json.load(fh)
                log.info("Loaded ETA model from %s", self.path)
                return self
            except Exception as exc:
                log.warning("Could not load saved ETA model (%s); retraining.", exc)
        X, y = generate_synthetic_dataset()
        self.model, self.metrics = train(X, y)
        try:
            os.makedirs(settings.models_dir, exist_ok=True)
            joblib.dump(self.model, self.path)
            with open(self.metrics_path, "w", encoding="utf-8") as fh:
                json.dump(self.metrics, fh, indent=2)
        except OSError as exc:
            log.warning("Could not save ETA model (%s); keeping it in memory only.", exc)
        return self

    def predict_minutes(self, cls: str, severity: int, distance_km: float = 5.0,
                        traffic_level: int = 1, rain_mm: float = 0.0) -> int:
        if self.model is None:
            self.load_or_train()
        x = np.array([[class_index(cls), severity, distance_km, traffic_level, rain_mm]], dtype=float)
        minutes = float(self.model.predict(x)[0])
        return int(max(10, round(minutes / 5.0) * 5))  # round to nearest 5 minutes
