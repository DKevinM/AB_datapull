import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values  # pip install psycopg2-binary

# --- Config ---
DEFAULT_HOURS_BACK = 6
UPSERT_CHUNK = 50_000
SLICE_HOURS = 6  # time-slice size for range backfills
DB_URL = os.environ.get("SUPABASE_DB_URL")

ODATA_STATIONS = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations?$select=Name"
ODATA_MEASUREMENTS = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"

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

# --- Station list ---
def fetch_station_names():
    r = requests.get(ODATA_STATIONS, timeout=30)
    r.raise_for_status()
    df = pd.json_normalize(r.json()["value"])[["Name"]]
    df["Name"] = df["Name"].astype(str).str.strip()
    return df["Name"].dropna().drop_duplicates().tolist()

# --- Time formatting for the API (use local Alberta offset, DST-safe) ---
def fmt_edmonton(ts_utc: datetime) -> str:
    """
    Convert a UTC-aware datetime to America/Edmonton with an explicit offset
    formatted like 2025-09-09T02:00:00-07:00 (or -06:00 in winter).
    """
    ts_local = pd.Timestamp(ts_utc).tz_convert("America/Edmonton")
    s = ts_local.strftime("%Y-%m-%dT%H:%M:%S%z")  # e.g., ...-0700
    return s[:-2] + ":" + s[-2:]                  # -> ...-07:00

# --- Fetch: last N hours (hourly mode) ---
def fetch_last_h(station_name: str, hours: int) -> pd.DataFrame:
    start_utc = datetime.now(timezone.utc) - timedelta(hours=hours)
    start_str = fmt_edmonton(start_utc)  # <— key change vs your old code
    safe = station_name.replace("'", "''")
    params = {
        "$format": "json",
        "$filter": f"StationName eq '{safe}' AND ReadingDate gt {start_str}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
    }
    try:
        r = requests.get(ODATA_MEASUREMENTS, params=params, timeout=45)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("value", []))
    except Exception as e:
        print(f"Fetch failed for {station_name!r}: {e}", file=sys.stderr)
        return pd.DataFrame()

# --- Fetch: explicit UTC range, sliced (manual backfill mode) ---
def fetch_slice(station_name: str, t0_utc: datetime, t1_utc: datetime) -> pd.DataFrame:
    assert t0_utc.tzinfo is timezone.utc and t1_utc.tzinfo is timezone.utc
    safe = station_name.replace("'", "''")
    start_str = fmt_edmonton(t0_utc)
    end_str   = fmt_edmonton(t1_utc)
    # closed-open to avoid dupes across slices
    flt = f"StationName eq '{safe}' AND ReadingDate ge {start_str} AND ReadingDate lt {end_str}"
    params = {
        "$format": "json",
        "$filter": flt,
        "$orderby": "ReadingDate asc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
    }
    try:
        r = requests.get(ODATA_MEASUREMENTS, params=params, timeout=60)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("value", []))
    except Exception as e:
        print(f"[WARN] slice fetch failed for {station_name!r} {start_str}..{end_str}: {e}", file=sys.stderr)
        return pd.DataFrame()

def fetch_between(station_name: str, start_utc: datetime, end_utc: datetime, slice_hours: int = SLICE_HOURS) -> pd.DataFrame:
    frames = []
    t0 = start_utc
    while t0 < end_utc:
        t1 = min(t0 + timedelta(hours=slice_hours), end_utc)
        df = fetch_slice(station_name, t0, t1)
        if not df.empty:
            frames.append(df)
        t0 = t1
        time.sleep(0.1)  # be polite
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["StationName","ParameterName","ReadingDate","Value"])

# --- Clean ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["StationName"] = out["StationName"].astype(str).str.strip()
    out["ParameterName"] = out["ParameterName"].fillna("").replace("", "AQHI")
    # API times are UTC; store as naive UTC TIMESTAMP
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

# --- CLI & main ---
def parse_utc(ts: str) -> datetime:
    # Accept '...Z' or with offset; normalize to UTC aware
    try:
        dt = pd.to_datetime(ts, utc=True).to_pydatetime()
        return dt.astimezone(timezone.utc)
    except Exception:
        print(f"ERROR parsing timestamp '{ts}'. Use ISO8601 like 2025-09-09T02:00:00Z", file=sys.stderr)
        sys.exit(2)

def build_argparser():
    p = argparse.ArgumentParser(description="Ingest last N hours (default) or a specific UTC time range.")
    p.add_argument("--hours-back", type=int, default=DEFAULT_HOURS_BACK,
                   help=f"Rolling window in hours (default {DEFAULT_HOURS_BACK}); ignored if start/end provided.")
    p.add_argument("--start-utc", type=str, help="Start (UTC), e.g. 2025-09-08T07:00:00Z")
    p.add_argument("--end-utc",   type=str, help="End (UTC, non-inclusive), e.g. 2025-09-10T07:00:00Z")
    p.add_argument("--slice-hours", type=int, default=SLICE_HOURS,
                   help=f"Slicing window for range fetch (default {SLICE_HOURS}).")
    p.add_argument("--station", action="append", default=None,
                   help="Limit to one or more StationName values (repeat --station). Default: all stations.")
    return p

def main():
    args = build_argparser().parse_args()
    engine = get_engine()
    ensure_aqhi_table(engine)

    # Station selection
    if args.station:
        stations = [s.strip() for s in args.station if s.strip()]
    else:
        stations = fetch_station_names()
    if not stations:
        print("No stations returned."); return

    # Mode A: explicit range
    if args.start_utc and args.end_utc:
        start_utc = parse_utc(args.start_utc)
        end_utc   = parse_utc(args.end_utc)
        if not (start_utc < end_utc):
            print("ERROR: start_utc must be earlier than end_utc", file=sys.stderr)
            sys.exit(2)

        total_in = total_sent = 0
        for name in stations:
            raw = fetch_between(name, start_utc, end_utc, slice_hours=args.slice_hours)
            if raw.empty:
                continue
            cleaned = clean_data(raw)
            total_in += len(cleaned)
            total_sent += upsert_measurements(engine, cleaned)
        print(f"✅ range={start_utc.isoformat()}..{end_utc.isoformat()} stations={len(stations)} "
              f"rows_in={total_in} rows_upserted={total_sent}")
        return

    # Mode B: rolling last N hours (default)
    frames = []
    for name in stations:
        df = fetch_last_h(name, hours=args.hours_back)
        if not df.empty:
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        cleaned = clean_data(combined)
        sent = upsert_measurements(engine, cleaned)
        print(f"✅ window={args.hours_back}h rows_upserted={sent} (input={len(cleaned)})")
    else:
        print("No data fetched in window.")

if __name__ == "__main__":
    main()
