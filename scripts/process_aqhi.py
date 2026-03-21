#!/usr/bin/env python3
"""
Process raw AQHI CSV → compute AQHI values → upsert to Supabase.

Usage:
    python scripts/process_aqhi.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import OUTPUT_DIR
from src.utils.logger import setup_logger
from src.processing.aqhi_processor import AQHIProcessor
from src.storage.supabase_handler import SupabaseHandler
from src.storage.geojson_writer import GeoJSONWriter

logger = setup_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Processing AQHI data...")

    raw_path = OUTPUT_DIR / "aqhi_raw.csv"
    if not raw_path.exists():
        logger.error(f"Raw AQHI file not found: {raw_path}. Run fetch_aqhi.py first.")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} raw rows from {raw_path}")

    processor = AQHIProcessor()
    processed = processor.process(df)
    logger.info(f"Processed {len(processed)} AQHI station records (mode: {processed['source_mode'].iloc[0] if not processed.empty else 'N/A'})")

    # Save processed CSV
    output_csv = OUTPUT_DIR / "aqhi_live.csv"
    processed.to_csv(output_csv, index=False)
    logger.info(f"Saved processed AQHI to {output_csv}")

    # Save GeoJSON
    writer = GeoJSONWriter(output_dir=OUTPUT_DIR)
    writer.write_sensor_map(processed, "aqhi_live.geojson", lat_col="Latitude", lon_col="Longitude")

    # Upsert to Supabase
    try:
        handler = SupabaseHandler()
        records = processed.to_dict("records")
        total = handler.upsert_sensor_readings(records)
        logger.info(f"Upserted {total} AQHI records to Supabase")
    except Exception as e:
        logger.error(f"Supabase upsert failed: {e}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
