"""
config.py
---------
Central configuration for VanSuraksha — Uttarakhand Forest Fire Risk System.

All tuneable parameters live here.  Import from this module; never hard-code
values in other files.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── Directory layout ──────────────────────────────────────────────────────────
INPUT_DIR   = BASE_DIR / "uk_fdata"          # Raw FIRMS CSV files go here
OUTPUT_DIR  = BASE_DIR / "outputs"           # Processed CSVs and reports
MODEL_DIR   = BASE_DIR / "models"            # Saved model artefacts
LOG_DIR     = BASE_DIR / "logs"

# Ensure output directories exist at import time
for _d in (INPUT_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Key file paths ────────────────────────────────────────────────────────────
CLEANED_CSV        = OUTPUT_DIR / "cleaned_fire_weather_data.csv"
TRAINING_CSV       = OUTPUT_DIR / "training_ready_data.csv"
WEATHER_CACHE_CSV  = OUTPUT_DIR / "weather_cache.csv"
MODEL_REPORT_CSV   = OUTPUT_DIR / "model_comparison_report.csv"
FEATURE_IMP_CSV    = MODEL_DIR  / "feature_importance.csv"

XGBOOST_PATH = MODEL_DIR / "xgboost_model.pkl"
RF_PATH      = MODEL_DIR / "rf_model.pkl"
SVM_PATH     = MODEL_DIR / "svm_model.pkl"
CNN_PATH     = MODEL_DIR / "cnn_model.h5"
SCALER_PATH  = MODEL_DIR / "feature_scaler.pkl"

# ── Uttarakhand geographic bounds ─────────────────────────────────────────────
BBOX = {
    "lat_min": 28.5,
    "lat_max": 31.5,
    "lon_min": 77.5,
    "lon_max": 81.0,
}
MAP_CENTER = {"lat": 30.0, "lon": 79.0, "zoom": 8}

# ── Open-Meteo weather variables ──────────────────────────────────────────────
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "soil_moisture_0_to_7cm",
]
WEATHER_PEAK_HOUR = 14          # 14:00 IST — peak fire-risk proxy hour
WEATHER_WORKERS   = 15          # Parallel fetch threads
WEATHER_CACHE_INTERVAL = 500    # Save cache every N successful fetches

# ── Feature columns — order must match exactly across all modules ─────────────
FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "THR",    # Temperature-Humidity Ratio
    "FSI",    # Fire Season Indicator
    "WTP",    # Wind-Temperature Product
    "SDI",    # Soil Dryness Index
    "year", "month", "day", "hour",
]

# Months considered peak fire season in Uttarakhand
FIRE_SEASON_MONTHS = [3, 4, 5, 6]

# ── Model training ────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CNN_EPOCHS   = 30
CNN_BATCH    = 64

XGBOOST_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)
RF_PARAMS = dict(
    n_estimators=300, class_weight="balanced",
    max_depth=12, random_state=RANDOM_STATE, n_jobs=-1,
)

# ── Risk level thresholds ─────────────────────────────────────────────────────
RISK_LEVELS = [
    (0.00, 0.20, "Minimal",  "#22c55e", "No immediate action required"),
    (0.20, 0.40, "Low",      "#84cc16", "Standard monitoring protocols"),
    (0.40, 0.60, "Moderate", "#f59e0b", "Heightened vigilance — resources on standby"),
    (0.60, 0.80, "High",     "#ef4444", "Active patrol deployment — alerts issued"),
    (0.80, 1.01, "Extreme",  "#7f1d1d", "Emergency response — immediate action required"),
]

# ── Flask server ──────────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE  = LOG_DIR / "vansuraksha.log"
LOG_LEVEL = "INFO"
