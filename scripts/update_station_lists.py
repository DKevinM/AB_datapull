#!/usr/bin/env python3
"""
Update AQHI station list and PurpleAir sensor list in Supabase.

Usage:
    python scripts/update_station_lists.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd
import geopandas as gpd

from config.settings import DATA_DIR, OUTPUT_DIR, SHAPEFILES, SENSOR_LISTS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

AQHI_STATIONS_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    "?$select=Name,Latitude,Longitude&$format=json&$top=1000"
)


def update_aqhi_stations() -> pd.DataFrame:
    """Fetch AQHI station list from Alberta API and save locally."""
    logger.info("Fetching AQHI station list...")
    r = requests.get(AQHI_STATIONS_URL, timeout=30)
    r.raise_for_status()
    df = pd.json_normalize(r.json().get("value", []))[["Name", "Latitude", "Longitude"]]
    df = df.dropna(subset=["Name", "Latitude", "Longitude"])
    df["Name"] = df["Name"].str.strip()
    df = df.drop_duplicates(subset=["Name"])

    # Save locally
    out_path = DATA_DIR / "station_list.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} AQHI stations to {out_path}")
    return df


def update_purpleair_sensors(province: str) -> pd.DataFrame:
    """Discover PurpleAir sensors for a province and update the local sensor list."""
    import os
    api_key = os.getenv("PURPLEAIR_API_KEY", "")
    if not api_key:
        logger.warning("PURPLEAIR_API_KEY not set; skipping PurpleAir sensor discovery")
        return pd.DataFrame()

    shapefile = SHAPEFILES.get(province)
    if not shapefile or not Path(shapefile).exists():
        logger.warning(f"Shapefile not found for province={province}: {shapefile}")
        return pd.DataFrame()

    boundary = gpd.read_file(str(shapefile))
    if boundary.crs is None or boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(epsg=4326)
    minx, miny, maxx, maxy = boundary.total_bounds

    url = "https://api.purpleair.com/v1/sensors"
    params = {
        "fields": "sensor_index,name,latitude,longitude,location_type,last_seen",
        "location_type": 0,
        "nwlng": minx,
        "nwlat": maxy,
        "selng": maxx,
        "selat": miny,
    }
    resp = requests.get(url, headers={"X-API-Key": api_key}, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data.get("data", []), columns=data.get("fields", []))

    if df.empty:
        logger.warning(f"No PurpleAir sensors found for province={province}")
        return df

    # Spatial clip
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    union = boundary.unary_union
    inside = gdf[gdf.geometry.intersects(union)].copy()
    inside["province"] = province

    # Save
    sensor_list_path = SENSOR_LISTS.get(province)
    if sensor_list_path:
        inside.drop(columns=["geometry"]).to_csv(str(sensor_list_path), index=False)
        logger.info(f"Updated sensor list: {len(inside)} sensors saved to {sensor_list_path}")

    return inside.drop(columns=["geometry"])


def push_stations_to_supabase(stations_df: pd.DataFrame):
    """Upsert AQHI stations to Supabase."""
    try:
        from src.storage.supabase_handler import SupabaseHandler
        handler = SupabaseHandler()
        records = stations_df.rename(columns={"Name": "StationName"}).to_dict("records")
        total = handler.upsert_stations(records)
        logger.info(f"Upserted {total} AQHI stations to Supabase")
    except Exception as e:
        logger.error(f"Failed to upsert stations to Supabase: {e}")


def push_sensors_to_supabase(sensors_df: pd.DataFrame, province: str):
    """Upsert PurpleAir sensor metadata to Supabase."""
    if sensors_df.empty:
        return
    try:
        from src.storage.supabase_handler import SupabaseHandler
        handler = SupabaseHandler()
        cols = [c for c in ["sensor_index", "name", "latitude", "longitude",
                             "location_type", "last_seen", "province"] if c in sensors_df.columns]
        records = sensors_df[cols].to_dict("records")
        total = handler.upsert_purpleair_sensors(records)
        logger.info(f"Upserted {total} PurpleAir sensors for {province} to Supabase")
    except Exception as e:
        logger.error(f"Failed to upsert PurpleAir sensors to Supabase: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Updating station and sensor lists...")

    # AQHI stations
    stations_df = update_aqhi_stations()
    push_stations_to_supabase(stations_df)

    # PurpleAir sensors per province
    for province in ["AB", "SK"]:
        sensors_df = update_purpleair_sensors(province)
        push_sensors_to_supabase(sensors_df, province)

    logger.info("Station and sensor lists updated.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
