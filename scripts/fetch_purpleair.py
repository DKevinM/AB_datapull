#!/usr/bin/env python3
"""
Pull PurpleAir sensors for a given province, filter to that province's boundary,
save the sensor metadata and push readings to Supabase.

Usage:
    python scripts/fetch_purpleair.py --province AB
    python scripts/fetch_purpleair.py --province SK
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd

from config.settings import OUTPUT_DIR, SHAPEFILES, SENSOR_LISTS, PURPLEAIR_MAX_DATA_AGE_HOURS
from src.utils.logger import setup_logger
from src.ingestion.purpleair_client import PurpleAirClient
from src.processing.pm25_processor import PM25Processor

logger = setup_logger(__name__)


def load_channel_override(province: str) -> dict:
    """Load channel-override CSV for the given province if it exists."""
    override_path = Path("data") / "channel_override.csv"
    if not override_path.exists():
        return {}
    try:
        df = pd.read_csv(override_path)
        df["sensor_index"] = pd.to_numeric(df["sensor_index"], errors="coerce")
        df = df.dropna(subset=["sensor_index"])
        df["sensor_index"] = df["sensor_index"].astype("int64")
        return dict(zip(df["sensor_index"], df["force_channel"]))
    except Exception as e:
        logger.warning(f"Could not read channel_override.csv: {e}")
        return {}


def load_dead_list() -> set:
    """Load sensor IDs that are known dead/offline."""
    dead_path = Path("data") / "dead_list.csv"
    if not dead_path.exists():
        return set()
    try:
        df = pd.read_csv(dead_path)
        df["sensor_index"] = pd.to_numeric(df["sensor_index"], errors="coerce")
        df = df.dropna(subset=["sensor_index"])
        return set(df["sensor_index"].astype("int64").tolist())
    except Exception as e:
        logger.warning(f"Could not read dead_list.csv: {e}")
        return set()


def infer_network(name: str, province: str) -> str:
    """Infer the airshed network label from sensor name."""
    n = name.upper()
    if "ACA" in n or "ALBERTA CAPITAL" in n:
        return "ACA"
    if "WCAS" in n or "WCA" in n:
        return "WCAS"
    if "PAS" in n:
        return "PAS"
    if "CRAZ" in n:
        return "CRAZ"
    return "OTHER"


def main():
    parser = argparse.ArgumentParser(description="Fetch PurpleAir data for a province.")
    parser.add_argument(
        "--province",
        choices=list(SHAPEFILES.keys()),
        default="AB",
        help="Province code (AB or SK)",
    )
    args = parser.parse_args()
    province = args.province

    shapefile = SHAPEFILES.get(province)
    if not shapefile or not Path(shapefile).exists():
        logger.error(f"No shapefile found for province={province}: {shapefile}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Fetching PurpleAir sensors for province={province}")

    # Load province boundary
    boundary = gpd.read_file(str(shapefile))
    if boundary.crs is None or boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(epsg=4326)
    minx, miny, maxx, maxy = boundary.total_bounds

    # Fetch from PurpleAir API using bounding box
    client = PurpleAirClient()
    df_raw = client.fetch_sensors_bbox(
        nwlng=minx, nwlat=maxy, selng=maxx, selat=miny
    )
    logger.info(f"API returned {len(df_raw)} sensors in bounding box")

    # Spatial clip to province polygon
    df_raw["latitude"] = pd.to_numeric(df_raw["latitude"], errors="coerce")
    df_raw["longitude"] = pd.to_numeric(df_raw["longitude"], errors="coerce")
    df_raw = df_raw.dropna(subset=["latitude", "longitude"])

    gdf = gpd.GeoDataFrame(
        df_raw,
        geometry=gpd.points_from_xy(df_raw.longitude, df_raw.latitude),
        crs="EPSG:4326",
    )
    province_union = boundary.unary_union
    inside = gdf[gdf.geometry.intersects(province_union)].copy()
    inside["province"] = province
    inside["network"] = inside["name"].apply(lambda n: infer_network(n, province))
    logger.info(f"{len(inside)} sensors inside {province} boundary")

    # Apply dead-list filter
    dead_ids = load_dead_list()
    if dead_ids:
        before = len(inside)
        inside["sensor_index"] = pd.to_numeric(inside["sensor_index"], errors="coerce")
        inside = inside[~inside["sensor_index"].isin(dead_ids)].copy()
        logger.info(f"Removed {before - len(inside)} dead sensors; {len(inside)} remaining")

    # Save sensor list CSV (metadata)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sensor_list_path = OUTPUT_DIR / f"purpleair_{province.lower()}_sensors.csv"
    meta_cols = [c for c in ["sensor_index", "name", "latitude", "longitude", "location_type",
                              "last_seen", "province", "network"] if c in inside.columns]
    inside[meta_cols].to_csv(sensor_list_path, index=False)
    logger.info(f"Saved sensor list to {sensor_list_path}")

    # Apply PM2.5 processing (channel selection + RH correction)
    channel_override = load_channel_override(province)
    processor = PM25Processor()

    # Filter out sensors seen >3 hours ago
    now_utc = datetime.now(timezone.utc)
    if "last_seen" in inside.columns:
        inside["last_seen"] = pd.to_numeric(inside["last_seen"], errors="coerce")
        inside["last_seen_dt"] = pd.to_datetime(inside["last_seen"], unit="s", utc=True)
        inside = inside[inside["last_seen_dt"] >= (now_utc - timedelta(hours=PURPLEAIR_MAX_DATA_AGE_HOURS))].copy()
        logger.info(f"{len(inside)} sensors with recent data (within {PURPLEAIR_MAX_DATA_AGE_HOURS}h)")

    pm_cols = ["pm2.5_atm", "pm2.5_atm_a", "pm2.5_atm_b", "humidity"]
    available_pm_cols = [c for c in pm_cols if c in inside.columns]
    if available_pm_cols:
        processed = processor.process_batch(inside, channel_override=channel_override)
    else:
        processed = inside.copy()
        logger.warning("PM2.5 columns not available; skipping channel selection")

    # Save processed output
    output_json = OUTPUT_DIR / f"purpleair_{province.lower()}_live.json"
    processed.drop(columns=["geometry"], errors="ignore").to_json(
        output_json, orient="records", indent=2
    )
    logger.info(f"Saved {len(processed)} processed sensors to {output_json}")

    # Push to Supabase
    try:
        from src.storage.supabase_handler import SupabaseHandler
        handler = SupabaseHandler()

        hourly_ts = now_utc.replace(minute=0, second=0, microsecond=0)
        records = []
        for _, row in processed.iterrows():
            records.append({
                "sensor_index": int(row["sensor_index"]) if pd.notna(row.get("sensor_index")) else None,
                "province": province,
                "recorded_at": hourly_ts.isoformat(),
                "pm25_atm_a": float(row["pm2.5_atm_a"]) if pd.notna(row.get("pm2.5_atm_a")) else None,
                "pm25_atm_b": float(row["pm2.5_atm_b"]) if pd.notna(row.get("pm2.5_atm_b")) else None,
                "humidity": float(row["humidity"]) if pd.notna(row.get("humidity")) else None,
                "pm_raw": float(row["pm_raw"]) if pd.notna(row.get("pm_raw")) else None,
                "pm_corrected": float(row["pm_corrected"]) if pd.notna(row.get("pm_corrected")) else None,
                "pm_method": str(row["pm_method"]) if pd.notna(row.get("pm_method")) else None,
            })
        records = [r for r in records if r["sensor_index"] is not None]
        total = handler.upsert_sensor_readings(records)
        logger.info(f"Upserted {total} sensor readings to Supabase")
    except Exception as e:
        logger.error(f"Supabase upsert failed: {e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

