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

## Setup (GitHub Actions — no local editor needed)

This project runs entirely through **GitHub Actions**. You do not need a code editor, Python installation, or `.env` file. Everything runs in the cloud automatically.

### 1. Add GitHub Actions secrets

Go to your repository on GitHub → **Settings → Secrets and variables → Actions → New repository secret**, and add these three secrets:

| Secret name | What to put in it |
|---|---|
| `SUPABASE_DB_URL` | Your Supabase project URL (e.g. `https://xxxx.supabase.co`) |
| `SUPABASE_SERVICE_KEY` | Your Supabase service role key (has write access) |
| `PURPLEAIR_API_KEY` | Your PurpleAir API read key |

Once the secrets are saved, the workflows will pick them up automatically — **no other configuration is needed**.

### 2. Enable the workflows

Go to the **Actions** tab in your repository. If workflows are disabled, click **"I understand my workflows, go ahead and enable them"**. The scheduled workflows will then run automatically on their set schedules (see the Workflows table below).

You can also run any workflow manually at any time: go to **Actions → pick a workflow → Run workflow**.

### 3. Creating new directories via the GitHub web UI

Git does not allow empty directories. To create a new folder directly on the GitHub website (without a code editor):

1. Navigate to the location in your repo where you want the new folder.
2. Click **Add file → Create new file**.
3. In the filename box, type the folder name followed by `/` and then a filename, for example:
   ```
   data/output/.gitkeep
   ```
   GitHub will automatically create the `data/output/` directory containing the `.gitkeep` file.
4. Scroll down and click **Commit new file**.

> **Note:** All directories used by the automated workflows (`data/output/`, `data/sensor_lists/`) are already pre-created in this repository with `.gitkeep` placeholder files, so you should not need to create them manually.

---

### Local development (optional)

If you want to run scripts locally from a terminal:

```bash
pip install -r requirements.txt

# Set secrets as environment variables, then run:
python scripts/fetch_aqhi.py --hours-back 24
python scripts/fetch_purpleair.py --province AB
python scripts/process_aqhi.py
python scripts/process_purpleair.py --province AB
python scripts/interpolate_grid.py --region AB
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
