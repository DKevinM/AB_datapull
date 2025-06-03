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
    """Hits the OData Stations endpoint and returns a DataFrame with column ['Name'].""" 
    resp = requests.get(STATIONS_ODATA_URL, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.json_normalize(raw["value"])
    return df[["Name"]]

stations_df = fetch_station_list()
print(f"Fetched {len(stations_df)} stations.")


# ───────────────────────────────────────────────────────────────
# 2. Define fetch_data_last24h(station_name)
# ───────────────────────────────────────────────────────────────

def fetch_last6h(station) -> pd.DataFrame:
    now = datetime.utcnow()
    start = now - timedelta(hours=6)
    # Format: YYYY-MM-DDTHH:MM:SS-06:00  (Alberta is UTC-6)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%S-06:00')
       
    url = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"
    params = {
        "$format": "json",
        "$filter": f"StationName eq '{station}' AND ReadingDate gt {start_str}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("value", [])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to fetch data for {station}: {e}")
        return pd.DataFrame()


# ───────────────────────────────────────────────────────────────
# 3. Loop over all stations, fetch & combine into one DataFrame
# ───────────────────────────────────────────────────────────────

Path("data").mkdir(exist_ok=True)
summary = []

for idx, row in stations_df.iterrows():
    station = row["Name"]
    df = fetch_last6h(station)
    clean_name = stn["StationName"].replace("’", "").replace("'", "").replace(" ", "_")
    if not df.empty:
        combined_rows.append(df)
    else:
        print(f"No data in the last 6h for {station!r}.")

if combined_rows:
    combined_df = pd.concat(combined_rows, ignore_index=True)
else:
    combined_df = pd.DataFrame(columns=["StationName", "ParameterName", "ReadingDate", "Value"])

# ───────────────────────────────────────────────────────────────
# 4. Write the combined DataFrame to a single CSV
# ───────────────────────────────────────────────────────────────

import os
cwd = os.getcwd()
print(f">>> Current working directory: {cwd}")
print(f">>> Total rows in combined_df: {len(combined_df)}")


output_folder = Path("data")
output_folder.mkdir(exist_ok=True)

combined_path = output_folder / "_last6h.csv"
print(f">>> Attempting to write CSV to: {combined_path}")
combined_df.to_csv(combined_path, index=False)

print(f"\nWrote combined data ({len(combined_df)} rows) to {combined_path}")
print(f">>> Finished writing CSV to: {combined_path}")
