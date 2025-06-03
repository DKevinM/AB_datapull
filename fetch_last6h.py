import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import os

# ───────────────────────────────────────────────────────────────
# 1. Fetch the full station list
# ───────────────────────────────────────────────────────────────

STATIONS_ODATA_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    "?$select=Name,Latitude,Longitude"
)

def fetch_station_list():
    resp = requests.get(STATIONS_ODATA_URL, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.json_normalize(raw["value"])
    return df[["Name", "Latitude", "Longitude"]]


# ───────────────────────────────────────────────────────────────
# 2. Define fetch_data_last24h(station_name)
# ───────────────────────────────────────────────────────────────

def fetch_last6h(station_name: str) -> pd.DataFrame:
    now = datetime.utcnow()
    start = now - timedelta(days=1)
    # Format: YYYY-MM-DDTHH:MM:SS-06:00  (Alberta is UTC-6)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%S-06:00')

    safe_name = station_name.replace("'", "''").replace("’", "''")
    
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

# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 3a. Fetch station list with raw names
    stations_df = fetch_station_list()
    print(f">>> Fetched {len(stations_df)} stations (raw names).")

    combined_rows = []

    for _, row in stations_df.iterrows():
        name = row["Name"]          # ← raw station name, possibly with apostrophes
        lat  = row["Latitude"]
        lon  = row["Longitude"]

        df = fetch_last6h(name)

        if not df.empty:
            # 3b. Attach Latitude and Longitude to each measurement row
            df["Latitude"]  = lat
            df["Longitude"] = lon

            combined_rows.append(df)
            print(f">>> Pulled {len(df)} rows for {name!r}.")
        else:
            print(f">>> No data in last 6h for {name!r}.")

    # 3c. Concatenate into a single DataFrame (with the exact columns)
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
    else:
        combined_df = pd.DataFrame(
            columns=["StationName", "ParameterName", "ReadingDate", "Value", "Latitude", "Longitude"]
        )



    # 3d. Debug prints
    cwd = os.getcwd()
    print(f">>> Current working directory: {cwd}")
    print(f">>> Total rows in combined_df: {len(combined_df)}")
    if not combined_df.empty:
        print(">>> Sample rows:")
        print(combined_df.head().to_string(index=False))

    # 3e. Write exactly one CSV to data/last6h.csv
    output_folder = Path("data")
    output_folder.mkdir(exist_ok=True)

    combined_path = output_folder / "last6h.csv"
    print(f">>> Attempting to write CSV to: {combined_path}")
    combined_df.to_csv(combined_path, index=False)
    print(f">>> Finished writing CSV ({len(combined_df)} rows) {combined_path}")
    # NEW: show exactly what’s in data/
    print(">>> Contents of data/ after writing:")
    for p in Path("data").iterdir():
        print("    ", p, "(exists)" if p.exists() else "(missing)")
