"""
app.py  —  VanSuraksha Flask Backend
--------------------------------------
Serves the interactive map UI and all REST API endpoints.

Endpoints
---------
GET  /                         → map.html dashboard
GET  /health                   → server + model status
POST /predict                  → live fire-risk prediction for a coordinate
POST /forecast                 → 7-day hourly risk forecast
GET  /hotspots                 → historical fire locations (filterable by year/month)
GET  /model-stats              → model comparison report
GET  /api/frp                  → FRP chart data (year + optional month)
GET  /api/monthly-summary      → per-month stats for a year
GET  /api/meta                 → available years and months
GET  /api/seasons              → FRP distribution by season
"""

from __future__ import annotations

import traceback

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import BBOX, FLASK_DEBUG, FLASK_HOST, FLASK_PORT, MAP_CENTER
from logger import logger
from services import (
    get_7day_forecast_risk,
    get_available_months,
    get_available_years,
    get_frp_by_year_month,
    get_hotspots,
    get_model_stats,
    get_monthly_summary,
    get_season_distribution,
    predict_fire_risk,
)
from services.predictor import models_ready

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

logger.info("=== VanSuraksha server starting ===")


# ── Static / template entry point ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main map dashboard."""
    return send_from_directory("templates", "map.html")


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """
    Returns server status and whether trained models are available on disk.
    The frontend uses models_ready to enable/disable the predict button.
    """
    return jsonify({
        "status":       "ok",
        "models_ready": models_ready(),
        "version":      "3.0",
        "region":       BBOX,
        "map_center":   MAP_CENTER,
    })


# ── Live prediction ───────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON body: { "lat": float, "lon": float }
    Returns risk probability, level, recommended action, and weather snapshot.
    """
    try:
        data = request.json or {}
        lat  = float(data.get("lat", MAP_CENTER["lat"]))
        lon  = float(data.get("lon", MAP_CENTER["lon"]))

        if not (BBOX["lat_min"] <= lat <= BBOX["lat_max"] and
                BBOX["lon_min"] <= lon <= BBOX["lon_max"]):
            return jsonify({
                "status":  "error",
                "message": "Coordinates are outside the Uttarakhand study area.",
            }), 400

        result = predict_fire_risk(lat, lon)
        return jsonify(result)

    except FileNotFoundError as exc:
        logger.error("Model artefact missing: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 503
    except Exception:
        logger.exception("Prediction error")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


# ── 7-day forecast ────────────────────────────────────────────────────────────

@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Accept JSON body: { "lat": float, "lon": float }
    Returns a list of 168 hourly { time, prob, level, color } entries.
    """
    try:
        data    = request.json or {}
        lat     = float(data.get("lat", MAP_CENTER["lat"]))
        lon     = float(data.get("lon", MAP_CENTER["lon"]))
        results = get_7day_forecast_risk(lat, lon)
        return jsonify(results)

    except Exception:
        logger.exception("Forecast error")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


# ── Historical hotspots ───────────────────────────────────────────────────────

@app.route("/hotspots")
def hotspots():
    """
    Returns high/nominal-confidence fire detections for the map heatmap layer.
    Optional query params: year (int), month (int)
    """
    try:
        year  = request.args.get("year",  type=int, default=None)
        month = request.args.get("month", type=int, default=None)
        data  = get_hotspots(year=year, month=month)
        return jsonify(data)

    except Exception:
        logger.exception("Hotspots error")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


# ── Model comparison stats ────────────────────────────────────────────────────

@app.route("/model-stats")
def model_stats():
    """Returns the per-model evaluation metrics from the training report CSV."""
    try:
        return jsonify(get_model_stats())
    except Exception:
        return jsonify({"status": "error", "message": "Stats unavailable"}), 500


# ── Historical data API — drives year/month dashboard ─────────────────────────

@app.route("/api/frp")
def api_frp():
    """
    Return FRP chart data.
    Required query param: year (int)
    Optional query param: month (int) — if omitted returns monthly aggregates
    """
    year  = request.args.get("year",  type=int)
    month = request.args.get("month", type=int, default=None)

    if year is None:
        return jsonify({"error": "year query parameter is required"}), 400

    return jsonify(get_frp_by_year_month(year, month))


@app.route("/api/monthly-summary")
def api_monthly_summary():
    """
    Return per-month fire statistics for a given year.
    Required query param: year (int)
    """
    year = request.args.get("year", type=int)
    if year is None:
        return jsonify({"error": "year query parameter is required"}), 400
    return jsonify(get_monthly_summary(year))


@app.route("/api/meta")
def api_meta():
    """
    Return available years and, if year is provided, available months.
    Used to populate the year/month selector dropdowns in the dashboard.
    """
    year   = request.args.get("year", type=int)
    years  = get_available_years()
    months = get_available_months(year) if year else []
    return jsonify({"years": years, "months": months})


@app.route("/api/seasons")
def api_seasons():
    """Return FRP value distributions grouped by season for box/violin charts."""
    return jsonify(get_season_distribution())


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
