import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
# 1. Fetch station list
# ─────────────────────────────────────────────

def fetch_station_list():
    STATIONS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations?$select=Name,Latitude,Longitude"
    resp = requests.get(STATIONS_URL, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    return pd.json_normalize(raw["value"])[["Name", "Latitude", "Longitude"]]


# ─────────────────────────────────────────────
# 2. Fetch last 15 days of data per station
# ─────────────────────────────────────────────

def fetch_last15d(station_name):
    now = datetime.utcnow()
    start = now - timedelta(days=15)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%S-06:00')  # Alberta time

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
    except Exception as e:
        print(f"Failed to fetch data for {station_name}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 3. Clean data
# ─────────────────────────────────────────────

def clean_data(df):
    df = df.copy()
    df["ParameterName"] = df["ParameterName"].replace('', 'AQHI')

    ppm_params = [
        "Nitric Oxide", "Nitrogen Dioxide", "Total Oxides of Nitrogen",
        "Sulphur Dioxide", "Ozone", "Carbon Monoxide"
    ]
    df.loc[df["ParameterName"].isin(ppm_params), "Value"] *= 1000

    return df


# ─────────────────────────────────────────────
# 4. Create DB connection
# ─────────────────────────────────────────────

def get_engine():
    DB_USER = os.environ["DB_DKEV"]
    DB_PASS = os.environ["DB_PASS"]
    DB_HOST = os.environ["DB_HOST"]
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ["DB_NAME"]
    return create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")



       
# ─────────────────────────────────────────────
# 5b. Efficient upsert via temp table
# ─────────────────────────────────────────────
def create_table_if_needed(engine):
    sql = """
    CREATE TABLE IF NOT EXISTS aqhi_data (
        StationName TEXT,
        ParameterName TEXT,
        ReadingDate TIMESTAMP,
        Value FLOAT,
        Latitude FLOAT,
        Longitude FLOAT,
        PRIMARY KEY (StationName, ParameterName, ReadingDate)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))  # use `text()` from sqlalchemy


def upsert_to_main_table(df, engine):
    with engine.begin() as conn:
        # 1. Create the temp table
        conn.execute("""
            CREATE TEMP TABLE temp_aqhi_data (
                StationName TEXT,
                ParameterName TEXT,
                ReadingDate TIMESTAMP,
                Value FLOAT,
                Latitude FLOAT,
                Longitude FLOAT
            );
        """)

        # 2. Load data into temp table
        df.to_sql("temp_aqhi_data", engine, if_exists="append", index=False, method='multi')

        # 3. Insert from temp into main with deduplication
        conn.execute("""
            INSERT INTO aqhi_data (StationName, ParameterName, ReadingDate, Value, Latitude, Longitude)
            SELECT StationName, ParameterName, ReadingDate, Value, Latitude, Longitude
            FROM temp_aqhi_data
            ON CONFLICT DO NOTHING;
        """)

# ─────────────────────────────────────────────
# 6. Run the whole pipeline
# ─────────────────────────────────────────────

def main():
    print(">>> Fetching station list...")
    stations = fetch_station_list()

    all_data = []
    for _, row in stations.iterrows():
        name = row["Name"]
        lat = row["Latitude"]
        lon = row["Longitude"]

        df = fetch_last15d(name)
        if df.empty:
            print(f"  No data for {name}")
            continue

        df["Latitude"] = lat
        df["Longitude"] = lon
        all_data.append(df)
        print(f"  Pulled {len(df)} rows from {name}")

    if not all_data:
        print(">>> No data found. Exiting.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    cleaned = clean_data(combined)

    print(f">>> Total cleaned rows: {len(cleaned)}")

    engine = get_engine()
    create_table_if_needed(engine)

    upsert_to_main_table(cleaned, engine)  # ← this is the new line
    print(">>> Data inserted into database (deduplicated).")


if __name__ == "__main__":
    main()
