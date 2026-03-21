#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import OUTPUT_DIR
from src.utils.logger import setup_logger
from src.ingestion.purpleair_client import PurpleAirClient

logger = setup_logger(__name__)

def main():
    logger.info("Fetching PurpleAir data...")
    
    # Load sensor list
    sensor_df = pd.read_csv("data/AB_PA_sensors.csv")
    sensor_ids = sensor_df['sensor_index'].dropna().astype(int).tolist()
    
    client = PurpleAirClient()
    df = client.fetch_sensors(sensor_ids)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "purpleair_raw.json"
    df.to_json(output_file, orient="records", indent=2)
    logger.info(f"✓ Saved {len(df)} sensors to {output_file}")

if __name__ == "__main__":
    main()
