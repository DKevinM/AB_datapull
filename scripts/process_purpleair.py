#!/usr/bin/env python3
"""
Process raw PurpleAir JSON → PM2.5 selection + RH correction → upsert to Supabase.

Usage:
    python scripts/process_purpleair.py --province AB
    python scripts/process_purpleair.py --province SK
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import OUTPUT_DIR
from src.utils.logger import setup_logger
from src.processing.pm25_processor import PM25Processor
from src.storage.geojson_writer import GeoJSONWriter

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process raw PurpleAir data for a province.")
    parser.add_argument("--province", choices=["AB", "SK"], default="AB")
    args = parser.parse_args()
    province = args.province

    logger.info("=" * 60)
    logger.info(f"Processing PurpleAir data for province={province}...")

    raw_path = OUTPUT_DIR / f"purpleair_{province.lower()}_live.json"
    if not raw_path.exists():
        logger.error(f"Raw PurpleAir file not found: {raw_path}. Run fetch_purpleair.py first.")
        sys.exit(1)

    df = pd.read_json(raw_path)
    logger.info(f"Loaded {len(df)} raw sensors from {raw_path}")

    # Channel selection + RH correction
    processor = PM25Processor()
    processed = processor.process_batch(df)
    logger.info(f"Processed {len(processed)} sensors")

    # Save GeoJSON for Leaflet
    writer = GeoJSONWriter(output_dir=OUTPUT_DIR)
    writer.write_sensor_map(processed, f"purpleair_{province.lower()}_live.geojson")

    # Upsert to Supabase
    try:
        from src.storage.supabase_handler import SupabaseHandler
        handler = SupabaseHandler()
        now_utc = datetime.now(timezone.utc)
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
        logger.info(f"Upserted {total} PurpleAir records to Supabase")
    except Exception as e:
        logger.error(f"Supabase upsert failed: {e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
