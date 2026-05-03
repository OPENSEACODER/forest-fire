# VanSuraksha — Uttarakhand Forest Fire Risk Intelligence

Production-level wildfire risk prediction and historical data exploration
built on NASA FIRMS fire detections, Open-Meteo weather data, XGBoost /
Random Forest / SVM / 1D CNN classifiers, and a Flask + Leaflet dashboard.

---

## Project layout

```
vansuraksha/
├── config.py               ← All tuneable parameters (single source of truth)
├── logger.py               ← Centralised logging (console + file)
├── app.py                  ← Flask server — all API routes
├── run_pipeline.py         ← One-shot CLI to run pipeline stages
│
├── pipeline/
│   ├── __init__.py
│   ├── data_loader.py      ← Stage 1: FIRMS CSV + Open-Meteo weather merge
│   ├── preprocessor.py     ← Stage 2: Feature engineering + labeling
│   └── trainer.py          ← Stage 4: XGBoost / RF / SVM / 1D CNN training
│
├── services/
│   ├── __init__.py
│   ├── predictor.py        ← Live fire-risk inference (XGBoost + live weather)
│   ├── forecast_engine.py  ← 7-day hourly forecast
│   └── data_service.py     ← Historical data queries → JSON for API
│
├── templates/
│   └── map.html            ← Full dashboard (Live Map + Historical Data tabs)
│
├── uk_fdata/               ← Place raw FIRMS CSV files here
├── outputs/                ← Cleaned CSVs, reports (auto-created)
├── models/                 ← Trained model artefacts (auto-created)
├── logs/                   ← Application logs (auto-created)
└── requirements.txt
```

---

## Quick start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Get FIRMS data
1. Go to https://firms.modaps.eosdis.nasa.gov/country/
2. Select: Country = **India**, Satellite = **VIIRS NOAA-20**, Date range = any
3. Download the CSV(s) and place them in `uk_fdata/`

### 3 — Run the pipeline
```bash
# Full pipeline (data → features → train)
python run_pipeline.py

# Or run individual stages:
python run_pipeline.py --stage 1   # fetch + merge
python run_pipeline.py --stage 2   # feature engineering
python run_pipeline.py --stage 4   # model training
```

### 4 — Start the server
```bash
python app.py
```
Open http://localhost:5000

---

## Dashboard features

### 🗺 Live Map tab
| Feature | Description |
|---|---|
| Click-to-predict | Click any point inside Uttarakhand to get live fire risk |
| Risk card | Probability, level (Minimal→Extreme), recommended action |
| Weather snapshot | Live temperature, humidity, wind, precipitation |
| 7-day forecast | Hourly risk probability chart for next 7 days |
| Hotspot layer | Toggle historical high-confidence fire detections |

### 📊 Historical Data tab
| Feature | Description |
|---|---|
| Year/Month selectors | Dynamically populated from available data |
| KPI cards | Total fire count, mean FRP, peak FRP, peak month |
| FRP trend chart | Monthly bar (yearly view) or daily line (monthly view) |
| Seasonal distribution | Median FRP + 75th percentile by season |
| Monthly summary table | Per-month fire count, mean/max FRP with peak highlighted |

---

## API reference

```
GET  /health                          Server + model status
POST /predict          {lat, lon}     Live fire risk prediction
POST /forecast         {lat, lon}     7-day hourly forecast
GET  /hotspots         ?year&month    Historical fire locations
GET  /model-stats                     Model comparison report
GET  /api/frp          ?year&month    FRP chart data
GET  /api/monthly-summary  ?year      Per-month statistics
GET  /api/meta         ?year          Available years / months
GET  /api/seasons                     FRP by season
```

---

## Configuration

All parameters are in `config.py`:

| Key | Default | Description |
|---|---|---|
| `BBOX` | Uttarakhand bounds | Geographic filter for fire data |
| `FIRE_SEASON_MONTHS` | `[3,4,5,6]` | Months flagged as peak fire season |
| `FEATURE_COLS` | 13 columns | Model feature order (never change without retraining) |
| `RISK_LEVELS` | 5 tiers | Probability thresholds → label + colour + action |
| `FLASK_PORT` | `5000` | Server port |
