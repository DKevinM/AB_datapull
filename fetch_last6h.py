import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ───────────────────────────────────────────────────────────────
# 1. Fetch the full station list (names only)
# ───────────────────────────────────────────────────────────────

STATIONS_ODATA_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    "?$select=Name"
)

def fetch_station_list():
    resp = requests.get(STATIONS_ODATA_URL, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.json_normalize(raw["value"])
    return df[["Name", "Latitude", "Longitude"]]


stations_df = fetch_station_list()
print(f"Fetched {len(stations_df)} stations.")


def make_safe_filename(name: str) -> str:
    safe = (
        name
        .replace("’", "")   # remove fancy apostrophes
        .replace("'", "")    # remove straight apostrophes
        .replace("/", "-")   # replace any forward slash with dash
        .replace(" ", "_")   # spaces → underscores
    )
    return safe


# ───────────────────────────────────────────────────────────────
# 2. Define fetch_data_last24h(station_name)
# ───────────────────────────────────────────────────────────────

def fetch_last6h(station_name) -> pd.DataFrame:
    now = datetime.utcnow()
    start = now - timedelta(hours=6)
    # Format: YYYY-MM-DDTHH:MM:SS-06:00  (Alberta is UTC-6)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%S-06:00')
       
    url = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"
    params = {
        "$format": "json",
        "$filter": f"StationName eq '{safe_name}' AND ReadingDate gt {start_str}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("value", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to fetch data for {station_name!r}: {e}")
        return pd.DataFrame()




OUT_BASE = Path("data") / "stations"
OUT_BASE.mkdir(parents=True, exist_ok=True)

summary_rows = []

for idx, row in stations_df.iterrows():
    station = row["Name"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    df = fetch_data(station)

    safe_fn = make_safe_filename(station)
    out_path = OUT_BASE / f"{safe_fn}.csv"

    if not df.empty:
        df.to_csv(out_path, index=False)
        # Get the latest ReadingDate in this DataFrame
        latest = pd.to_datetime(df["ReadingDate"], errors="coerce").max()
        print(f"Wrote {len(df)} rows → {out_path}  (last: {latest})")
        summary_rows.append({
            "StationName": station,
            "Latitude": lat,
            "Longitude": lon,
            "LastReading": latest
        })
    else:
        if out_path.exists():
            print(f"No new data for {station!r}; keeping existing {out_path}.")
            # Optionally, read the existing CSV’s last date:
            try:
                old = pd.read_csv(out_path, parse_dates=["ReadingDate"])
                old_max = old["ReadingDate"].max() if "ReadingDate" in old.columns else pd.NaT
            except Exception:
                old_max = pd.NaT
            summary_rows.append({
                "StationName": station,
                "Latitude": lat,
                "Longitude": lon,
                "LastReading": old_max
            })
        else:
            print(f"No data at all for {station!r}. No file created.")
            summary_rows.append({
                "StationName": station,
                "Latitude": lat,
                "Longitude": lon,
                "LastReading": pd.NaT
            })


if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_out = Path("data") / "stations_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"\nWrote summary → {summary_out}")

