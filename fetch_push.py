import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# ------------------------------
# 3) Measurements fetch (no lat/lon)
# ------------------------------
def fetch_last_d(station_name, days=1, start_time=None):
    if start_time is None:
        start_time = datetime.utcnow()
    start = start_time - timedelta(days=days)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC (Zulu)
    safe_name = station_name.replace("'", "''")

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
        return pd.DataFrame(r.json().get("value", []))
    except Exception:
        return pd.DataFrame()


# ------------------------------
# 4) Clean measurements
# ------------------------------
def clean_data(df):
    df = df.copy()
    if df.empty:
        return df
    df["ParameterName"] = df["ParameterName"].fillna("").replace("", "AQHI")
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], utc=True)
    ppm_params = [
        "Nitric Oxide", "Nitrogen Dioxide", "Total Oxides of Nitrogen",
        "Sulphur Dioxide", "Ozone", "Carbon Monoxide"
    ]
    df.loc[df["ParameterName"].isin(ppm_params), "Value"] = df.loc[df["ParameterName"].isin(ppm_params), "Value"] * 1000
    # Remove known outliers
    df = df[~((df["ParameterName"] == "Ozone") & (df["Value"] > 150))]
    # Remove invalid values
    df = df[df["Value"].notna()]
    df = df.drop_duplicates(subset=["StationName", "ParameterName", "ReadingDate"])
    return df



# ------------------------------
# 5) DB helpers
# ------------------------------
def get_engine():
    return create_engine(os.environ["SUPABASE_DB_URL"])

def create_measurements_table_if_needed(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.aqhi_data (
          station_id BIGINT NOT NULL,
          "ParameterName" TEXT NOT NULL,
          "ReadingDate" TIMESTAMP NOT NULL,
          "Value" DOUBLE PRECISION,
          PRIMARY KEY (station_id, "ParameterName", "ReadingDate"),
          FOREIGN KEY (station_id) REFERENCES public.stations(station_id)
            ON UPDATE CASCADE ON DELETE RESTRICT
        );
        """))

def get_station_id_map(engine) -> dict:
    with engine.begin() as conn:
        res = conn.execute(text("SELECT station_id, station_name FROM public.stations;"))
        return {row.station_name: row.station_id for row in res}

def upsert_measurements(engine, df: pd.DataFrame, name_to_id: dict):
    if df.empty:
        return
    # Map StationName -> station_id, drop rows we can't map (should be none if you just synced stations)
    df = df[df["StationName"].isin(name_to_id.keys())].copy()
    if df.empty:
        return
    df["station_id"] = df["StationName"].map(name_to_id)
    df = df.rename(columns={"ParameterName": "ParameterName", "ReadingDate": "ReadingDate", "Value": "Value"})
    df = df[["station_id", "ParameterName", "ReadingDate", "Value"]]

    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TEMP TABLE tmp_aqhi_data (
          station_id BIGINT NOT NULL,
          "ParameterName" TEXT NOT NULL,
          "ReadingDate" TIMESTAMP NOT NULL,
          "Value" DOUBLE PRECISION,
          PRIMARY KEY (station_id, "ParameterName", "ReadingDate")
        ) ON COMMIT DROP;
        """))
        df.to_sql("tmp_aqhi_data", conn, if_exists="append", index=False, method="multi")
        conn.execute(text("""
        INSERT INTO public.aqhi_data (station_id, "ParameterName", "ReadingDate", "Value")
        SELECT station_id, "ParameterName", "ReadingDate", "Value"
        FROM tmp_aqhi_data
        ON CONFLICT (station_id, "ParameterName", "ReadingDate")
        DO UPDATE SET "Value" = EXCLUDED."Value"
        WHERE EXCLUDED."Value" IS NOT NULL;
        """))



# ------------------------------
# 6) Orchestration
# ------------------------------
def main(days_back=1):
    engine = get_engine()

    # A) Stations: fetch + upsert (do this once per run; schedule the script daily if you want)
    stations_df = fetch_station_list()
    upsert_stations(engine, stations_df)
    name_to_id = get_station_id_map(engine)

    # B) Measurements: create table, then fetch per station (no lat/lon), clean, map station_id, upsert
    create_measurements_table_if_needed(engine)

    all_data = []
    for station_name in stations_df["station_name"]:
        df = fetch_last_d(station_name, days=days_back)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return

    combined = pd.concat(all_data, ignore_index=True)
    cleaned = clean_data(combined)
    upsert_measurements(engine, cleaned, name_to_id)

if __name__ == "__main__":
    # run daily with days_back=1; or pass a larger window when you need a backfill
    main(days_back=1)
    
