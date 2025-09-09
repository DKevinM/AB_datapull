import os
import sys
import math
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values  # pip install psycopg2-binary

# --- Config ---
HOURS_BACK = 6
UPSERT_CHUNK = 50_000
DB_URL = os.environ.get("SUPABASE_DB_URL")

ODATA_STATIONS = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations?$select=Name"
)
ODATA_MEASUREMENTS = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"
)

PPM_PARAMS = {
    "Nitric Oxide",
    "Nitrogen Dioxide",
    "Total Oxides of Nitrogen",
    "Sulphur Dioxide",
    "Ozone",
    "Carbon Monoxide",
}

# --- DB helpers ---
def get_engine():
    if not DB_URL:
        print("ERROR: SUPABASE_DB_URL env var not set", file=sys.stderr)
        sys.exit(1)
    return create_engine(DB_URL)

def ensure_aqhi_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS public.aqhi_data (
      "StationName"   TEXT NOT NULL,
      "ParameterName" TEXT NOT NULL,
      "ReadingDate"   TIMESTAMP NOT NULL,
      "Value"         DOUBLE PRECISION,
      PRIMARY KEY ("StationName","ParameterName","ReadingDate")
    );
    CREATE INDEX IF NOT EXISTS aqhi_readingdate_idx ON public.aqhi_data ("ReadingDate");
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

# --- Fetch ---
def fetch_station_names():
    r = requests.get(ODATA_STATIONS, timeout=30)
    r.raise_for_status()
    df = pd.json_normalize(r.json()["value"])[["Name"]]
    df["Name"] = df["Name"].astype(str).str.strip()
    return df["Name"].dropna().drop_duplicates().tolist()

def fetch_last_h(station_name: str, hours: int = HOURS_BACK) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")  # UTC (Zulu)
    safe = station_name.replace("'", "''")

    params = {
        "$format": "json",
        "$filter": f"StationName eq '{safe}' AND ReadingDate gt {start_str}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
    }
    try:
        r = requests.get(ODATA_MEASUREMENTS, params=params, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("value", []))
    except Exception as e:
        print(f"Fetch failed for {station_name!r}: {e}", file=sys.stderr)
        return pd.DataFrame()

# --- Clean ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    out["StationName"] = out["StationName"].astype(str).str.strip()
    out["ParameterName"] = out["ParameterName"].fillna("").replace("", "AQHI")
    out["ReadingDate"] = pd.to_datetime(out["ReadingDate"], utc=True).dt.tz_convert(None)

    # ppm -> ppb for common gases
    ppm_mask = out["ParameterName"].isin(PPM_PARAMS)
    out.loc[ppm_mask, "Value"] = out.loc[ppm_mask, "Value"] * 1000

    # known outliers
    out = out[~((out["ParameterName"] == "Ozone") & (out["Value"] > 150))]

    # drop invalids & dups
    out = out.dropna(subset=["StationName", "ParameterName", "ReadingDate", "Value"])
    out = out.drop_duplicates(subset=["StationName", "ParameterName", "ReadingDate"])

    return out[["StationName", "ParameterName", "ReadingDate", "Value"]]

# --- Upsert ---
def upsert_measurements(engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = list(df.itertuples(index=False, name=None))
    sql = """
      INSERT INTO public.aqhi_data ("StationName","ParameterName","ReadingDate","Value")
      VALUES %s
      ON CONFLICT ("StationName","ParameterName","ReadingDate")
      DO UPDATE SET "Value" = EXCLUDED."Value"
      -- only write when value actually changed
      WHERE public.aqhi_data."Value" IS DISTINCT FROM EXCLUDED."Value";
    """
    sent = 0
    with engine.begin() as sa_conn:
        pg = sa_conn.connection
        with pg.cursor() as cur:
            for i in range(0, len(rows), UPSERT_CHUNK):
                execute_values(cur, sql, rows[i:i+UPSERT_CHUNK], page_size=20_000)
                sent += min(UPSERT_CHUNK, len(rows) - i)
    return sent

# --- Main ---
def main(hours_back: int = HOURS_BACK):
    engine = get_engine()
    ensure_aqhi_table(engine)

    stations = fetch_station_names()
    if not stations:
        print("No stations returned."); return

    frames = []
    for name in stations:
        df = fetch_last_h(name, hours=hours_back)
        if not df.empty:
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        cleaned = clean_data(combined)
        sent = upsert_measurements(engine, cleaned)
        print(f"✅ window={hours_back}h rows_upserted={sent} (input={len(cleaned)})")
    else:
        print("No data fetched in window.")

if __name__ == "__main__":
    main(hours_back=6)
