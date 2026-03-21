# AB_datapull — Live Air Quality Data Pipeline

A modular data pipeline that pulls air quality data from the **Alberta AQHI government API** and **PurpleAir sensors**, processes it, and stores it in **Supabase** for live mapping.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              GitHub Actions (6 workflows)        │
│  01-update-station-lists  (daily)               │
│  02-fetch-live-data       (every 30 mins)       │
│  03-process-live-data     (triggered by 02)     │
│  04-generate-grids        (hourly)              │
│  05-health-check          (every 30 mins)       │
│  06-cleanup-artifacts     (daily)               │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  AQHI OData API            PurpleAir API v1
  (Alberta Govt)            (Single province pull)
        │                           │
        ▼                           ▼
  scripts/fetch_aqhi.py   scripts/fetch_purpleair.py
        │                     --province AB|SK
        └──────────┬────────────────┘
                   ▼
          data/output/ (raw CSVs + JSON)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  process_aqhi.py    process_purpleair.py
  (AQHI formula)     (PM2.5 channel select
                      + RH correction)
        │                     │
        └──────────┬──────────┘
                   ▼
             Supabase DB
         (sensor_readings table)
                   │
                   ▼
       interpolate_grid.py
       (IDW for AB/ACA/PAZA/PAS/SK)
                   │
                   ▼
         data/output/*_grid.geojson
         (for Leaflet heatmaps)
```

---

## Folder Structure

```
AB_datapull/
├── .github/workflows/          # 6 automated workflows
├── config/
│   └── settings.py             # Central config (paths, thresholds, env vars)
├── data/
│   ├── *.shp                   # Provincial/airshed boundary shapefiles
│   ├── AB_PA_sensors.csv       # Alberta PurpleAir sensor list
│   ├── dead_list.csv           # Known offline sensors
│   ├── channel_override.csv    # Manual A/B channel overrides
│   ├── sensor_lists/           # Generated sensor lists
│   └── output/                 # Generated CSVs, GeoJSONs, IDW grids
├── dataSK/
│   ├── SK.shp                  # Saskatchewan boundary
│   └── SK_PA_sensors.csv       # SK PurpleAir sensor list
├── scripts/                    # Entry-point scripts (called by workflows)
│   ├── fetch_aqhi.py           # Pull AQHI measurements
│   ├── fetch_purpleair.py      # Pull PurpleAir by province (AB or SK)
│   ├── process_aqhi.py         # Compute AQHI → Supabase
│   ├── process_purpleair.py    # PM2.5 processing → Supabase
│   ├── interpolate_grid.py     # IDW grid generation by region
│   ├── update_station_lists.py # Daily station/sensor discovery
│   ├── health_check.py         # Data freshness monitoring
│   ├── build_eAQHI.py          # Estimated AQHI (gas + PurpleAir PM2.5)
│   └── historical_to_db.py     # Backfill historical data
└── src/                        # Shared library
    ├── ingestion/
    │   ├── base_client.py      # HTTP client with retry logic
    │   ├── aqhi_client.py      # Alberta AQHI OData client
    │   └── purpleair_client.py # PurpleAir API client
    ├── processing/
    │   ├── aqhi_processor.py   # AQHI formula / unlabeled rows
    │   ├── pm25_processor.py   # Dual-channel PM2.5 + RH correction
    │   ├── geospatial.py       # IDW interpolation with cKDTree
    │   └── validators.py       # Outlier detection, freshness checks
    ├── storage/
    │   ├── supabase_handler.py # Batch upserts to Supabase
    │   ├── csv_writer.py       # Write CSVs to data/output/
    │   └── geojson_writer.py   # Write GeoJSON to data/output/
    ├── api/
    │   └── queries.py          # Read-only Supabase queries
    └── utils/
        ├── logger.py           # Structured logging
        └── exceptions.py       # Custom exception types
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required secrets (also set as GitHub Actions secrets):

| Variable | Description |
|---|---|
| `SUPABASE_DB_URL` | Your Supabase project URL (`https://xxxx.supabase.co`) |
| `SUPABASE_SERVICE_KEY` | Service role key (write access) |
| `PURPLEAIR_API_KEY` | PurpleAir API read key |

### 3. Run manually

```bash
# Fetch latest AQHI data
python scripts/fetch_aqhi.py --hours-back 24

# Fetch PurpleAir for Alberta (single province pull)
python scripts/fetch_purpleair.py --province AB

# Process and push to Supabase
python scripts/process_aqhi.py
python scripts/process_purpleair.py --province AB

# Generate interpolation grids
python scripts/interpolate_grid.py --region AB
python scripts/interpolate_grid.py --region ACA

# Health check
python scripts/health_check.py
```

---

## Workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `01-update-station-lists` | Daily midnight | Discover new AQHI stations + PurpleAir sensors |
| `02-fetch-live-data` | Every 30 mins | Pull latest AQHI + PurpleAir data |
| `03-process-live-data` | After `02` completes | Process, compute AQHI, upsert to Supabase |
| `04-generate-grids` | Hourly at :15 | IDW interpolation for AB, ACA, PAZA, PAS, SK |
| `05-health-check` | Every 30 mins | Alert if data is stale (>45 mins) |
| `06-cleanup-artifacts` | Daily 1 AM | Delete artifacts older than 7 days |

---

## Supabase Tables

| Table | Description |
|---|---|
| `stations` | AQHI station metadata (name, lat, lon) |
| `purpleair_sensors_meta` | PurpleAir sensor metadata |
| `sensor_readings` | Live readings (upsert key: `sensor_index, province, recorded_at`) |
