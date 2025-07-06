import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 1. Fetch station list
def fetch_station_list():
    STATIONS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations?$select=Name,Latitude,Longitude"
    resp = requests.get(STATIONS_URL, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    return pd.json_normalize(raw["value"])[["Name", "Latitude", "Longitude"]]

# 2. Fetch data per station
def fetch_last_d(station_name, days=7, start_time=None):
    """
    Fetch AQHI data for the past `days` from `start_time` (UTC).
    If no `start_time` is provided, defaults to `datetime.utcnow()`.
    """
    if start_time is None:
        start_time = datetime.utcnow()

    start = start_time - timedelta(days=days)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')  # UTC time (Z = Zulu = UTC)

    safe_name = station_name.replace("'", "''")  # escape apostrophes
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


# 3. Clean data
def clean_data(df):
    df = df.copy()
    df["ParameterName"] = df["ParameterName"].fillna("").replace('', 'AQHI')
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], utc=True)
    ppm_params = [
        "Nitric Oxide", "Nitrogen Dioxide", "Total Oxides of Nitrogen",
        "Sulphur Dioxide", "Ozone", "Carbon Monoxide"
    ]
    df.loc[df["ParameterName"].isin(ppm_params), "Value"] *= 1000
    # Remove known outliers
    df = df[~((df["ParameterName"] == "Ozone") & (df["Value"] > 150))]

    # Remove invalid values
    df = df[df["Value"].notna()]  # drop rows with null values
    df = df.drop_duplicates(subset=["StationName", "ParameterName", "ReadingDate"])
    return df


# 4. Create DB connection
def get_engine():
    return create_engine(os.environ["SUPABASE_DB_URL"])

       
# 5. Efficient upsert via temp table
def create_table_if_needed(engine):
    sql = """
    CREATE TABLE IF NOT EXISTS aqhi_data (
        "StationName" TEXT NOT NULL,
        "ParameterName" TEXT NOT NULL,
        "ReadingDate" TIMESTAMP NOT NULL,
        "Value" DOUBLE PRECISION,
        "Latitude" DOUBLE PRECISION,
        "Longitude" DOUBLE PRECISION,
        PRIMARY KEY ("StationName", "ParameterName", "ReadingDate")
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))  # use `text()` from sqlalchemy


def upsert_to_main_table(df, engine):
    # Replace null/blank ParameterName with "AQHI"
    df["ParameterName"] = df["ParameterName"].fillna("AQHI")
    df["ParameterName"] = df["ParameterName"].replace("", "AQHI")
    
    with engine.begin() as conn:
        # Step 1: Create temp table
        conn.execute(text("""
        CREATE TEMP TABLE temp_aqhi_data (
            "StationName" TEXT NOT NULL,
            "ParameterName" TEXT NOT NULL,
            "ReadingDate" TIMESTAMP NOT NULL,
            "Value" DOUBLE PRECISION,
            "Latitude" DOUBLE PRECISION,
            "Longitude" DOUBLE PRECISION,
            PRIMARY KEY ("StationName", "ParameterName", "ReadingDate")
            );
        """))

        # Step 2: Insert into temp table (bulk insert)
        df = df[["StationName", "ParameterName", "ReadingDate", "Value", "Latitude", "Longitude"]]
        df.to_sql("temp_aqhi_data", con=conn, if_exists="append", index=False, method='multi')

        # Step 3: Insert into main table with deduplication
        conn.execute(text("""
            INSERT INTO aqhi_data ("StationName", "ParameterName", "ReadingDate", "Value", "Latitude", "Longitude")
            SELECT "StationName", "ParameterName", "ReadingDate", "Value", "Latitude", "Longitude"
            FROM temp_aqhi_data
            ON CONFLICT ("StationName", "ParameterName", "ReadingDate") DO UPDATE
            SET "Value" = EXCLUDED."Value",
                "Latitude" = EXCLUDED."Latitude",
                "Longitude" = EXCLUDED."Longitude"
            WHERE EXCLUDED."Value" IS NOT NULL;
        """))


# 6. Run the whole pipeline

def main():
    stations = fetch_station_list()

    all_data = []
    for _, row in stations.iterrows():
        name = row["Name"]
        lat = row["Latitude"]
        lon = row["Longitude"]

        df = fetch_last_d(name)
        if df.empty:
            continue

        df["Latitude"] = lat
        df["Longitude"] = lon
        all_data.append(df)

    if not all_data:
        return
    
    non_empty = [df for df in all_data if not df.empty]
    combined = pd.concat(non_empty, ignore_index=True)
    cleaned = clean_data(combined)

    engine = get_engine()
    create_table_if_needed(engine)

    upsert_to_main_table(cleaned, engine)


if __name__ == "__main__":
    main()
