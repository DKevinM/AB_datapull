import os
import sys
import requests
import pandas as pd
from urllib.parse import urlparse
from sqlalchemy import create_engine, text

# Optional: use psycopg2 execute_values for fast bulk upserts
from psycopg2.extras import execute_values

def get_engine():
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        print("ERROR: SUPABASE_DB_URL env var is not set.", file=sys.stderr)
        sys.exit(1)
    return create_engine(url)

def fetch_station_list():
    url = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations?$select=Name,Latitude,Longitude"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    raw = r.json()
    df = pd.json_normalize(raw["value"])[["Name", "Latitude", "Longitude"]]

    # clean
    df = df.dropna(subset=["Name", "Latitude", "Longitude"])
    df["Name"] = df["Name"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["Name"])

    # rename to EXACT DB column names
    df = df.rename(columns={"Name": "StationName"})
    return df[["StationName", "Latitude", "Longitude"]]

def ensure_table(engine):
    """Create stations table if missing (NO PostGIS)."""
    sql = """
    CREATE TABLE IF NOT EXISTS public.stations (
      "StationName" TEXT PRIMARY KEY,
      "Latitude"    DOUBLE PRECISION NOT NULL,
      "Longitude"   DOUBLE PRECISION NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))

def upsert_stations(engine, df: pd.DataFrame):
    """Bulk UPSERT (ON CONFLICT) into public.stations using psycopg2 execute_values."""
    if df.empty:
        print("No stations to upsert.")
        return

    # Convert to list-of-tuples
    rows = list(df.itertuples(index=False, name=None))  # (StationName, Latitude, Longitude)

    # Build SQL once
    sql = """
        INSERT INTO public.stations ("StationName","Latitude","Longitude")
        VALUES %s
        ON CONFLICT ("StationName") DO UPDATE
        SET "Latitude" = EXCLUDED."Latitude",
            "Longitude" = EXCLUDED."Longitude";
    """

    # Use the raw DBAPI connection underneath SQLAlchemy for execute_values
    with engine.begin() as sa_conn:
        raw = sa_conn.connection  # psycopg2 connection
        with raw.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)

def main():
    try:
        engine = get_engine()
        ensure_table(engine)
        df = fetch_station_list()
        upsert_stations(engine, df)
        print(f"Upserted {len(df)} stations.")
    except requests.RequestException as e:
        print(f"Network error fetching station list: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
