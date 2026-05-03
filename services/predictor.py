"""
services/predictor.py
----------------------
Handles live fire-risk prediction for a given lat/lon.

Flow:
  1. Load trained XGBoost model + scaler from disk (cached after first load).
  2. Fetch current weather from Open-Meteo forecast API.
  3. Engineer features identical to training pipeline.
  4. Run model inference → probability + risk level + action recommendation.

Original source: predict.py
Changes: restructured into class-based service with LRU model cache;
         logging; paths and constants from config.py; docstrings added.
"""

from __future__ import annotations

import functools
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests

from config import FEATURE_COLS, FIRE_SEASON_MONTHS, MODEL_DIR, RISK_LEVELS, SCALER_PATH, XGBOOST_PATH

# Path to the optimal classification threshold saved by trainer.py
THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.txt"

def _load_threshold() -> float:
    """
    Load the optimal probability threshold found during training.
    Falls back to 0.55 (slightly above naive 0.5) if file not found.
    A threshold tuned on the real class distribution prevents the model
    from flagging every location as high-risk.
    """
    try:
        return float(THRESHOLD_PATH.read_text().strip())
    except Exception:
        logger.warning("Threshold file not found — using default 0.55")
        return 0.55
from logger import logger


# ── Model / scaler loader (loaded once, cached for server lifetime) ────────────

@functools.lru_cache(maxsize=1)
def _load_assets():
    """
    Load the XGBoost model and StandardScaler from disk.
    Results are cached so the files are only read once per process.
    Raises FileNotFoundError if either artefact is missing.
    """
    if not XGBOOST_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{XGBOOST_PATH}'. "
            "Run pipeline/trainer.py first."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'. "
            "Run pipeline/trainer.py first."
        )
    model  = joblib.load(XGBOOST_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Loaded model from %s", XGBOOST_PATH)
    return model, scaler


# ── Public API ────────────────────────────────────────────────────────────────

def predict_fire_risk(lat: float, lon: float) -> dict:
    """
    Main prediction entry point.

    Parameters
    ----------
    lat, lon : float — WGS-84 coordinates inside Uttarakhand

    Returns
    -------
    dict with keys: status, timestamp, location,
                    risk_metrics, weather_snapshot
    """
    model, scaler = _load_assets()
    weather       = _fetch_live_weather(lat, lon)
    dt            = datetime.now()

    features   = _build_features(weather, dt)
    X_scaled   = scaler.transform(features)
    raw_prob   = float(model.predict_proba(X_scaled)[:, 1][0])
    threshold  = _load_threshold()

    # Re-scale the raw probability relative to the optimal threshold.
    # Without this, a model trained on imbalanced data outputs 90%+ everywhere.
    if raw_prob < threshold:
        prob = raw_prob * (0.4 / max(threshold, 0.01))
    else:
        prob = 0.4 + (raw_prob - threshold) * (0.6 / max(1.0 - threshold, 0.01))
    prob = round(min(max(prob, 0.0), 1.0), 4)
    risk = _get_risk_level(prob)

    logger.info(
        "Prediction lat=%.4f lon=%.4f raw=%.3f threshold=%.2f scaled=%.3f level=%s",
        lat, lon, raw_prob, threshold, prob, risk["label"],
    )

    return {
        "status":    "success",
        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "location":  {"lat": round(lat, 4), "lon": round(lon, 4)},
        "risk_metrics": {
            "probability":     round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "level":  risk["label"],
            "color":  risk["color"],
            "action": risk["action"],
        },
        "weather_snapshot": {
            "temperature_c":    round(weather["temp"],   1),
            "humidity_pct":     round(weather["hum"],    1),
            "wind_kmh":         round(weather["wind"],   1),
            "precipitation_mm": round(weather["precip"], 2),
            "soil_moisture":    round(weather["soil"],   3),
        },
    }


def models_ready() -> bool:
    """Return True if both model artefacts exist on disk."""
    return XGBOOST_PATH.exists() and SCALER_PATH.exists()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_live_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather from Open-Meteo forecast API.
    Falls back to safe defaults on any network / parse error.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&hourly=soil_moisture_0_to_7cm"
        f"&timezone=Asia%2FKolkata"
        f"&forecast_days=1"
    )
    _defaults = {"temp": 25.0, "hum": 50.0, "wind": 5.0, "precip": 0.0, "soil": 0.25}

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Weather API error (%s) — using fallback values", exc)
        return _defaults

    current    = data.get("current", {})
    hourly_sm  = data.get("hourly", {}).get("soil_moisture_0_to_7cm", [])
    soil       = next((v for v in hourly_sm if v is not None), 0.25)

    return {
        "temp":   current.get("temperature_2m",       25.0) or 25.0,
        "hum":    current.get("relative_humidity_2m",  50.0) or 50.0,
        "wind":   current.get("wind_speed_10m",          5.0) or 5.0,
        "precip": current.get("precipitation",           0.0) or 0.0,
        "soil":   soil if soil is not None else 0.25,
    }


def _build_features(w: dict, dt: datetime) -> pd.DataFrame:
    """
    Construct the 13-column feature DataFrame that exactly matches training.
    Engineering formulae must stay in sync with pipeline/preprocessor.py.
    """
    soil = max(w["soil"], 1e-6)
    hum  = max(w["hum"],  0.0)

    row = {
        "temperature_2m":          w["temp"],
        "relative_humidity_2m":    hum,
        "wind_speed_10m":          w["wind"],
        "precipitation":           w["precip"],
        "soil_moisture_0_to_7cm":  soil,
        "THR": w["temp"] / (hum + 1),
        "FSI": 1 if dt.month in FIRE_SEASON_MONTHS else 0,
        "WTP": w["wind"] * w["temp"],
        "SDI": w["temp"] / (soil + 1e-6),
        "year":  dt.year,
        "month": dt.month,
        "day":   dt.day,
        "hour":  dt.hour,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def _get_risk_level(prob: float) -> dict:
    """Map a [0,1] probability to a labelled risk tier."""
    for lo, hi, label, color, action in RISK_LEVELS:
        if lo <= prob < hi:
            return {"label": label, "color": color, "action": action}
    return {"label": "Extreme", "color": "#7f1d1d",
            "action": "Emergency response — immediate action required"}