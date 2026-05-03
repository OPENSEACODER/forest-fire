"""
pipeline/data_loader.py  —  Stage 1
------------------------------------
Loads all FIRMS CSV exports from uk_fdata/, filters to Uttarakhand bounds,
fetches historical weather from the Open-Meteo Archive API in parallel,
then merges and saves cleaned_fire_weather_data.csv.

Run standalone:
    python -m pipeline.data_loader

Original source: 1_pipeline.py (Science Buddies / VanSuraksha)
Changes: restructured into importable module; logging replaces print; paths
         from config.py; docstrings added; no logic changes.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    BBOX, CLEANED_CSV, INPUT_DIR, WEATHER_CACHE_CSV,
    WEATHER_PEAK_HOUR, WEATHER_VARS, WEATHER_WORKERS, WEATHER_CACHE_INTERVAL,
)
from logger import logger


# ── Public entry point ────────────────────────────────────────────────────────

def run(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    """
    Full Stage-1 pipeline.

    1. Load and concatenate all *.csv files from input_dir.
    2. Filter rows to Uttarakhand bounding box.
    3. Fetch historical weather for each unique (date, grid) pair.
    4. Inner-join fire detections with weather data.
    5. Save merged result to CLEANED_CSV.

    Returns the merged DataFrame.
    Raises FileNotFoundError if no CSVs are found in input_dir.
    """
    _check_input_dir(input_dir)

    fire_df          = _load_fire_data(input_dir)
    unique_contexts  = _unique_grid_points(fire_df)
    weather_df       = _fetch_weather_parallel(unique_contexts)
    merged           = _merge_and_save(fire_df, weather_df)

    return merged


# ── Step helpers ──────────────────────────────────────────────────────────────

def _check_input_dir(path: Path) -> None:
    """Raise if the input directory has no CSV files."""
    csvs = list(path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in '{path}'. "
            "Download FIRMS data from https://firms.modaps.eosdis.nasa.gov/country/ "
            "and place the CSV exports in uk_fdata/."
        )
    logger.info("Found %d CSV file(s) in %s", len(csvs), path)


def _load_fire_data(path: Path) -> pd.DataFrame:
    """
    Concatenate all CSVs, filter to Uttarakhand bbox,
    and round coordinates to 0.1° grid.
    """
    all_files = list(path.glob("*.csv"))
    fire_df = pd.concat(
        [pd.read_csv(f) for f in all_files],
        ignore_index=True,
    )

    # Geographic filter
    mask = (
        fire_df["latitude"].between(BBOX["lat_min"], BBOX["lat_max"]) &
        fire_df["longitude"].between(BBOX["lon_min"], BBOX["lon_max"])
    )
    fire_df = fire_df[mask].copy()
    fire_df["date"]      = pd.to_datetime(fire_df["acq_date"])
    fire_df["lat_grid"]  = fire_df["latitude"].round(1)
    fire_df["lon_grid"]  = fire_df["longitude"].round(1)

    logger.info("Loaded %d fire detections inside Uttarakhand bounds", len(fire_df))
    return fire_df


def _unique_grid_points(fire_df: pd.DataFrame) -> pd.DataFrame:
    """Return the unique (date, lat_grid, lon_grid) combinations."""
    unique = fire_df.groupby(["date", "lat_grid", "lon_grid"]).size().reset_index()
    logger.info(
        "Fire points: %d  |  Unique (date × grid) pairs: %d",
        len(fire_df), len(unique),
    )
    return unique


def _fetch_weather_parallel(unique_contexts: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch one-day weather snapshots for each unique context in parallel.
    Caches progress to WEATHER_CACHE_CSV every WEATHER_CACHE_INTERVAL fetches.
    Returns a DataFrame with one row per successful fetch.
    """
    logger.info("Fetching weather for %d contexts (workers=%d)…",
                len(unique_contexts), WEATHER_WORKERS)

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=WEATHER_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_one, row): row
            for _, row in unique_contexts.iterrows()
        }

        for future in tqdm(as_completed(futures),
                           total=len(futures), desc="Weather fetch"):
            result = future.result()
            if result:
                results.append(result)

            # Periodic checkpoint so long runs don't lose work
            if len(results) % WEATHER_CACHE_INTERVAL == 0 and results:
                pd.DataFrame(results).to_csv(WEATHER_CACHE_CSV, index=False)

    if not results:
        raise RuntimeError(
            "No weather data was fetched. "
            "Check internet connectivity and Open-Meteo API availability."
        )

    logger.info("Weather fetch complete: %d/%d rows succeeded",
                len(results), len(unique_contexts))
    return pd.DataFrame(results)


def _fetch_one(row: pd.Series) -> dict | None:
    """
    Fetch a single date × location snapshot from the Open-Meteo Archive API.
    Returns a dict on success, None on any failure (network, timeout, etc.).
    Peak fire-risk hour (14:00 IST, index 14) is used for daily snapshot.
    """
    d_str = row["date"].strftime("%Y-%m-%d")
    lat, lon = row["lat_grid"], row["lon_grid"]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={d_str}&end_date={d_str}"
        f"&hourly={','.join(WEATHER_VARS)}&timezone=Asia/Kolkata"
    )

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data   = json.loads(resp.read())
            hourly = data["hourly"]
            row_data = {var: hourly[var][WEATHER_PEAK_HOUR] for var in WEATHER_VARS}
            row_data.update({"date": row["date"], "lat_grid": lat, "lon_grid": lon})
            return row_data
    except Exception:
        return None


def _merge_and_save(fire_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join fire detections with their matching weather data,
    save to CLEANED_CSV, and return the merged DataFrame.
    """
    weather_df["date"]    = pd.to_datetime(weather_df["date"])
    fire_df["date"]       = pd.to_datetime(fire_df["date"])

    for df in (fire_df, weather_df):
        df["lat_grid"] = df["lat_grid"].round(1)
        df["lon_grid"] = df["lon_grid"].round(1)

    merged = pd.merge(fire_df, weather_df, on=["date", "lat_grid", "lon_grid"], how="inner")
    merged.to_csv(CLEANED_CSV, index=False)
    logger.info("Saved %d merged records → %s", len(merged), CLEANED_CSV)
    return merged


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    run()
    logger.info("Stage 1 complete in %.2f min", (time.time() - t0) / 60)
