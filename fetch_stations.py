import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text

# --- 0) DB connection ---
def get_engine():
    # e.g. postgres://postgres:<PW>@db.<ref>.supabase.co:5432/postgres?sslmode=require
    return create_engine(os.environ["SUPABASE_DB_URL"])

# --- 1) Fetch station list from Alberta OData ---
def fetch_station_list():
    url = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations?$select=Name,Latitude,Longitude"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.json_normalize(r.json()["value"])[["Name", "Latitude", "Longitude"]]
    df = df.dropna(subset=["Name", "Latitude", "Longitude"]).drop_duplicates(subset=["Name"])
    df["Name"] = df["Name"].str.strip()
    return df.rename(columns={"Name": "station_name"})

# --- 2) Upsert into Supabase (daily) ---
def upsert_stations(engine, stations_df: pd.DataFrame):
    with engine.begin() as conn:
        # Enable PostGIS once
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))

        # Create table if needed (idempotent)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.stations (
          station_name TEXT NOT NULL UNIQUE,
          latitude     DOUBLE PRECISION NOT NULL,
          longitude    DOUBLE PRECISION NOT NULL,
          geom         geometry(Point,4326)
            GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude),4326)) STORED
        );
        """))

        # Temp table for fast bulk insert
        conn.execute(text("""
        CREATE TEMP TABLE tmp_stations (
          station_name TEXT,
          latitude     DOUBLE PRECISION,
          longitude    DOUBLE PRECISION
        ) ON COMMIT DROP;
        """))

        stations_df.to_sql("tmp_stations", conn, if_exists="append", index=False, method="multi")

        # Upsert by station_name; if coords change, update them
        conn.execute(text("""
        INSERT INTO public.stations (station_name, latitude, longitude)
        SELECT station_name, latitude, longitude
        FROM tmp_stations
        ON CONFLICT (station_name) DO UPDATE
          SET latitude  = EXCLUDED.latitude,
              longitude = EXCLUDED.longitude;
        """))

# --- 3) Orchestrate (call once every 24h) ---
def main():
    engine = get_engine()
    stations = fetch_station_list()
    if not stations.empty:
        upsert_stations(engine, stations)

if __name__ == "__main__":
    main()
