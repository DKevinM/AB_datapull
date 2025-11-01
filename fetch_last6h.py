import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import os
import time
from zoneinfo import ZoneInfo

STATIONS_ODATA_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    "?$select=Name,Latitude,Longitude"
)

MEAS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"

AB_TZ = ZoneInfo("America/Edmonton")   # robust across DST
DEFAULT_TIMEOUT = 45
RETRIES = 3

def http_get(url, *, params=None, timeout=DEFAULT_TIMEOUT, tries=RETRIES):
    """Simple retry with linear backoff on read/connect timeouts."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout) as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
        except requests.HTTPError:
            # No point retrying 4xx except maybe 429; keep it simple here.
            raise

def fetch_station_list() -> pd.DataFrame:
    resp = http_get(STATIONS_ODATA_URL, timeout=30)
    raw = resp.json()
    df = pd.json_normalize(raw["value"])
    return df[["Name", "Latitude", "Longitude"]]

def fetch_last6h(station_name: str) -> pd.DataFrame:
    # Compute the last 6h in **Alberta time** (handles DST correctly)
    now_ab = datetime.now(AB_TZ)
    start_ab = now_ab - timedelta(hours=6)

    # OData safest literal is datetimeoffset'YYYY-MM-DDTHH:MM:SS±HH:MM'
    start_literal = f"datetimeoffset'{start_ab.strftime('%Y-%m-%dT%H:%M:%S%z')[:-2]}:{start_ab.strftime('%Y-%m-%dT%H:%M:%S%z')[-2:]}'"
    # Example -> datetimeoffset'2025-11-01T17:00:00-06:00'

    safe_name = station_name.replace("'", "''")

    params = {
        "$format": "json",
        "$filter": f"StationName eq '{safe_name}' AND ReadingDate gt {start_literal}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
        "$top": "5000",  # avoid paging for busy stations
    }

    try:
        r = http_get(MEAS_URL, params=params, timeout=DEFAULT_TIMEOUT)
        data = r.json().get("value", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # enforce predictable dtypes
        return df.astype({
            "StationName": "string",
            "ParameterName": "string",
            "ReadingDate": "string",   # keep as string for CSV output parity
            "Value": "float64",
        }, errors="ignore")
    except Exception as e:
        print(f"Failed to fetch data for {station_name!r}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    stations_df = fetch_station_list()
    print(f">>> Fetched {len(stations_df)} stations (raw names).")

    combined_rows = []

    for _, row in stations_df.iterrows():
        name = row["Name"]
        lat  = row["Latitude"]
        lon  = row["Longitude"]

        df = fetch_last6h(name)
        if not df.empty:
            # Attach lat/lon
            df["Latitude"]  = float(lat)
            df["Longitude"] = float(lon)
            combined_rows.append(df)
            print(f">>> Pulled {len(df)} rows for {name!r}.")
        else:
            print(f">>> No data in last 6h for {name!r}.")

    # Filter out empties before concat (silences FutureWarning)
    usable = [d for d in combined_rows if d is not None and not d.empty]
    if usable:
        combined_df = pd.concat(usable, ignore_index=True)
        # Stable column order
        combined_df = combined_df[
            ["Value", "StationName", "ParameterName", "ReadingDate", "Latitude", "Longitude"]
        ]
    else:
        combined_df = pd.DataFrame(
            columns=["Value", "StationName", "ParameterName", "ReadingDate", "Latitude", "Longitude"]
        )

    cwd = os.getcwd()
    print(f">>> Current working directory: {cwd}")
    print(f">>> Total rows in combined_df: {len(combined_df)}")
    if not combined_df.empty:
        print(">>> Sample rows:")
        print(combined_df.head().to_string(index=False))

    # Write CSV
    output_folder = Path("data")
    output_folder.mkdir(exist_ok=True)
    combined_path = output_folder / "last6h.csv"
    print(f">>> Attempting to write CSV to: {combined_path}")
    combined_df.to_csv(combined_path, index=False)
    print(f">>> Finished writing CSV ({len(combined_df)} rows) {combined_path}")

    print(">>> Contents of data/ after writing:")
    for p in sorted(Path("data").iterdir(), key=lambda x: x.name.lower()):
        print("    ", p, "(exists)" if p.exists() else "(missing)")
