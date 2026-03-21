#!/usr/bin/env python3
"""
Fetch AQHI stations & measurements, write raw CSV and GeoJSON.

Usage:
    python scripts/fetch_aqhi.py [--hours-back 24]
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import OUTPUT_DIR, AQHI_LOOKBACK_HOURS
from src.utils.logger import setup_logger
from src.ingestion.aqhi_client import AQHIClient

logger = setup_logger(__name__)

SLEEP_BETWEEN = 0.20  # seconds between station requests (throttle)


def build_argparser():
    p = argparse.ArgumentParser(description="Fetch AQHI measurements for all stations.")
    p.add_argument(
        "--hours-back",
        type=int,
        default=AQHI_LOOKBACK_HOURS,
        help=f"Lookback window in hours (default: {AQHI_LOOKBACK_HOURS})",
    )
    return p


def main():
    args = build_argparser().parse_args()
    hours_back = args.hours_back

    logger.info("=" * 60)
    logger.info(f"Starting AQHI data fetch (last {hours_back}h)...")

    client = AQHIClient()
    stations_df = client.fetch_stations()
    logger.info(f"Found {len(stations_df)} stations")

    all_measurements = []
    for _, row in stations_df.iterrows():
        station_name = row["Name"]
        try:
            meas_df = client.fetch_measurements(station_name, hours_back=hours_back)
            if not meas_df.empty:
                meas_df["Latitude"] = float(row["Latitude"])
                meas_df["Longitude"] = float(row["Longitude"])
                all_measurements.append(meas_df)
        except Exception as e:
            logger.warning(f"Failed to fetch {station_name!r}: {e}")
        time.sleep(SLEEP_BETWEEN)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if all_measurements:
        combined_df = pd.concat(all_measurements, ignore_index=True)
        output_file = OUTPUT_DIR / "aqhi_raw.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(combined_df)} records to {output_file}")
    else:
        logger.warning("No AQHI data fetched")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

