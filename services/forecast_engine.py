"""
services/forecast_engine.py
-----------------------------
Generates 168-hour (7-day) fire-risk probability forecasts for any
Uttarakhand coordinate by combining Open-Meteo forecast weather with the
trained XGBoost model.

Original source: forecast_engine.py
Changes: restructured; logging; paths/constants from config; docstrings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import pandas as pd
import requests

from config import FEATURE_COLS, FIRE_SEASON_MONTHS, RISK_LEVELS, SCALER_PATH, XGBOOST_PATH
from logger import logger


# ── Public API ────────────────────────────────────────────────────────────────

def get_7day_forecast_risk(lat: float, lon: float) -> list[dict[str, Any]]:
    """
    Produce 168 hourly fire-risk predictions (7 days × 24 h) for lat/lon.

    Returns
    -------
    List of dicts, each with keys: time, prob, level, color.
    Returns an empty list on API error.
    """
    hourly = _fetch_forecast_weather(lat, lon)
    if not hourly:
        return []

    model  = joblib.load(XGBOOST_PATH)
    scaler = joblib.load(SCALER_PATH)

    rows = [_build_row(hourly, i) for i in range(len(hourly["time"]))]
    df   = pd.DataFrame(rows)[FEATURE_COLS]

    X_scaled = scaler.transform(df)
    probs    = model.predict_proba(X_scaled)[:, 1]

    logger.info("7-day forecast generated: %d hours  lat=%.4f lon=%.4f",
                len(probs), lat, lon)
    return [_format_hour(hourly["time"][i], float(probs[i])) for i in range(len(probs))]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_forecast_weather(lat: float, lon: float) -> dict | None:
    """
    Fetch 7-day hourly forecast from Open-Meteo.
    Returns the 'hourly' sub-dict on success, None on failure.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"precipitation,soil_moisture_0_to_7cm"
        f"&timezone=Asia%2FKolkata"
        f"&forecast_days=7"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()["hourly"]
    except Exception as exc:
        logger.warning("Forecast API error: %s", exc)
        return None


def _build_row(hourly: dict, i: int) -> dict:
    """Build a single feature row for hour index i."""
    t    = hourly["temperature_2m"][i]       or 20.0
    rh   = hourly["relative_humidity_2m"][i] or 50.0
    ws   = hourly["wind_speed_10m"][i]        or 5.0
    pr   = hourly["precipitation"][i]         or 0.0
    sm   = hourly["soil_moisture_0_to_7cm"][i]or 0.25
    dt   = datetime.fromisoformat(hourly["time"][i])
    soil = max(sm, 1e-6)

    return {
        "temperature_2m":          t,
        "relative_humidity_2m":    rh,
        "wind_speed_10m":          ws,
        "precipitation":           pr,
        "soil_moisture_0_to_7cm":  soil,
        "THR": t  / (rh + 1),
        "FSI": 1 if dt.month in FIRE_SEASON_MONTHS else 0,
        "WTP": ws * t,
        "SDI": t  / (soil + 1e-6),
        "year": dt.year, "month": dt.month,
        "day":  dt.day,  "hour":  dt.hour,
    }


def _format_hour(time_str: str, prob: float) -> dict[str, Any]:
    """Map a raw probability to a labelled, coloured risk entry."""
    for lo, hi, label, color, _ in RISK_LEVELS:
        if lo <= prob < hi:
            return {"time": time_str, "prob": round(prob, 3),
                    "level": label, "color": color}
    return {"time": time_str, "prob": round(prob, 3),
            "level": "Extreme", "color": "#7f1d1d"}
