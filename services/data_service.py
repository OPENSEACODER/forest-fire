"""
services/data_service.py
-------------------------
Read-only query layer over the processed fire-weather CSV.
All functions return JSON-serialisable Python structures consumed by
Flask API routes.

This module owns ALL data-access logic for the historical dashboard.
It is intentionally stateless — the caller passes the DataFrame in or
the module loads it from disk on first call.
"""

from __future__ import annotations

import functools
from typing import Any

import pandas as pd

from config import CLEANED_CSV, FIRE_SEASON_MONTHS, MODEL_REPORT_CSV
from logger import logger


# ── In-process data cache ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_fire_df() -> pd.DataFrame | None:
    """
    Load cleaned_fire_weather_data.csv once and cache it for the process lifetime.
    Returns None (not raises) if the file doesn't exist so callers can return
    graceful empty responses.
    """
    if not CLEANED_CSV.exists():
        logger.warning("Cleaned CSV not found at %s", CLEANED_CSV)
        return None
    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    logger.info("Data service loaded %d rows from %s", len(df), CLEANED_CSV)
    return df


# ── Map hotspots ──────────────────────────────────────────────────────────────

def get_hotspots(
    year: int | None = None,
    month: int | None = None,
    limit: int = 2000,
) -> list[dict]:
    """
    Return high/nominal-confidence fire hotspots for the Leaflet map layer.

    Parameters
    ----------
    year, month : optional filters (None = all available data)
    limit       : max rows returned (caps for frontend performance)
    """
    df = _load_fire_df()
    if df is None:
        return []

    sub = _filter_by_date(df, year, month)
    sub = sub[sub["confidence"].isin(["h", "n"])].copy()
    sub = sub.tail(limit)

    cols = ["latitude", "longitude", "date", "frp", "confidence"]
    cols = [c for c in cols if c in sub.columns]
    sub  = sub[cols].copy()

    if "frp" in sub.columns:
        sub["frp"] = sub["frp"].round(2)
    if "date" in sub.columns:
        sub["date"] = sub["date"].astype(str)

    return sub.to_dict(orient="records")


# ── FRP time-series (year/month selectable) ───────────────────────────────────

def get_frp_by_year_month(
    year: int,
    month: int | None = None,
) -> dict[str, Any]:
    """
    Return FRP values aggregated for a year (monthly bar chart)
    or a specific month (daily line chart).

    Returns
    -------
    dict with keys: chart_type, title, labels, values,
                    avg_frp, max_frp, fire_count
    """
    df = _load_fire_df()
    if df is None:
        return _empty_chart(year, month)

    sub = _filter_by_date(df, year, month)
    if sub.empty:
        return _empty_chart(year, month)

    if month is None:
        # Monthly mean FRP — bar chart
        monthly = sub.resample("ME", on="date")["frp"].mean().dropna()
        return {
            "chart_type": "bar",
            "title":      f"Monthly Mean FRP — {year}",
            "labels":     monthly.index.strftime("%b").tolist(),
            "values":     monthly.round(3).tolist(),
            "avg_frp":    round(float(sub["frp"].mean()), 2),
            "max_frp":    round(float(sub["frp"].max()), 2),
            "fire_count": int(sub["confidence"].isin(["h", "n"]).sum()),
        }
    else:
        # Daily FRP — line chart
        daily = sub.set_index("date").resample("D")["frp"].mean().dropna()
        month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B")
        return {
            "chart_type": "line",
            "title":      f"Daily Mean FRP — {month_name} {year}",
            "labels":     daily.index.strftime("%Y-%m-%d").tolist(),
            "values":     daily.round(3).tolist(),
            "avg_frp":    round(float(sub["frp"].mean()), 2),
            "max_frp":    round(float(sub["frp"].max()), 2),
            "fire_count": int(sub["confidence"].isin(["h", "n"]).sum()),
        }


def get_monthly_summary(year: int) -> list[dict]:
    """
    Return a per-month breakdown of fire count, mean FRP, and max FRP
    for a given year.  Used by the summary table in the dashboard.
    """
    df = _load_fire_df()
    if df is None:
        return []

    sub = _filter_by_date(df, year)
    if sub.empty:
        return []

    sub = sub[sub["confidence"].isin(["h", "n"])].copy()
    grp = sub.groupby(sub["date"].dt.month)["frp"].agg(
        fire_count="count",
        mean_frp="mean",
        max_frp="max",
    ).reset_index()
    grp.columns = ["month", "fire_count", "mean_frp", "max_frp"]
    grp["month_name"] = grp["month"].apply(
        lambda m: pd.Timestamp(year=year, month=m, day=1).strftime("%B")
    )
    grp["mean_frp"] = grp["mean_frp"].round(2)
    grp["max_frp"]  = grp["max_frp"].round(2)
    return grp.to_dict(orient="records")


# ── Metadata ──────────────────────────────────────────────────────────────────

def get_available_years() -> list[int]:
    """Return sorted list of years present in the cleaned dataset."""
    df = _load_fire_df()
    if df is None:
        return []
    return sorted(df["date"].dt.year.unique().tolist())


def get_available_months(year: int) -> list[int]:
    """Return sorted list of months available for a given year."""
    df = _load_fire_df()
    if df is None:
        return []
    sub = df[df["date"].dt.year == year]
    return sorted(sub["date"].dt.month.unique().tolist())


def get_season_distribution() -> dict[str, list[float]]:
    """
    Return FRP values grouped by season for box-plot / violin rendering.

    Uttarakhand season mapping (approximate):
        Spring : Mar–May   (peak fire season)
        Summer : Jun–Aug
        Autumn : Sep–Nov
        Winter : Dec–Feb
    """
    df = _load_fire_df()
    if df is None:
        return {}

    season_map = {3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn",
                  12: "Winter", 1: "Winter", 2: "Winter"}

    df = df.copy()
    df["season"] = df["date"].dt.month.map(season_map)

    result = {}
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        vals = df.loc[df["season"] == season, "frp"].dropna().round(3).tolist()
        result[season] = vals

    return result


# ── Model stats ───────────────────────────────────────────────────────────────

def get_model_stats() -> list[dict]:
    """Return the model comparison report rows."""
    if not MODEL_REPORT_CSV.exists():
        return []
    return pd.read_csv(MODEL_REPORT_CSV).to_dict(orient="records")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _filter_by_date(
    df: pd.DataFrame,
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    if year is not None:
        df = df[df["date"].dt.year == year]
    if month is not None:
        df = df[df["date"].dt.month == month]
    return df.copy()


def _empty_chart(year: int, month: int | None) -> dict:
    title = f"No data for {month or ''}/{year}".strip("/")
    return {
        "chart_type": "bar", "title": title,
        "labels": [], "values": [],
        "avg_frp": 0, "max_frp": 0, "fire_count": 0,
    }
