#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import OUTPUT_DIR
from src.utils.logger import setup_logger
from src.ingestion.aqhi_client import AQHIClient

logger = setup_logger(__name__)

def main():
    """Fetch AQHI stations and measurements"""
    logger.info("=" * 60)
    logger.info("Starting AQHI data fetch...")
    
    client = AQHIClient()
    stations_df = client.fetch_stations()
    
    all_measurements = []
    for idx, row in stations_df.iterrows():
        station_name = row['Name']
        try:
            meas_df = client.fetch_measurements(station_name)
            if not meas_df.empty:
                meas_df['Latitude'] = row['Latitude']
                meas_df['Longitude'] = row['Longitude']
                all_measurements.append(meas_df)
        except Exception as e:
            logger.warning(f"Failed to fetch {station_name}: {e}")
    
    if all_measurements:
        combined_df = pd.concat(all_measurements, ignore_index=True)
        output_file = OUTPUT_DIR / "aqhi_raw.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved {len(combined_df)} records to {output_file}")
    else:
        logger.warning("No data fetched")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()


# scripts/fetch_aqhi.py
"""
Pull AQHI stations & measurements, store raw data
Usage: python scripts/fetch_aqhi.py --hours-back 24 --output-csv
"""
def main():
    client = AQHIClient()
    stations = client.fetch_stations()
    
    all_measurements = []
    for station in stations:
        measurements = client.fetch_measurements(station['Name'], hours_back=24)
        all_measurements.append(measurements)
    
    df = pd.concat(all_measurements)
    
    # Save raw
    df.to_csv("data/output/aqhi_raw.csv", index=False)
    logger.info(f"Fetched {len(df)} AQHI measurements")

# scripts/fetch_purpleair.py
"""
Pull PurpleAir sensors for AB + SK
Usage: python scripts/fetch_purpleair.py --region AB
"""
def main(region="AB"):
    client = PurpleAirClient()
    sensor_list = pd.read_csv(f"data/sensor_lists/{region.lower()}_pa_sensors.csv")
    sensor_ids = sensor_list['sensor_index'].tolist()
    
    df = client.fetch_sensors(region, sensor_ids)
    df.to_json(f"data/output/purpleair_{region}_raw.json", orient="records")
    logger.info(f"Fetched {len(df)} PurpleAir sensors for {region}")

# scripts/process_aqhi.py
"""
Process raw AQHI → compute AQHI if needed → store in Supabase
Usage: python scripts/process_aqhi.py
"""
def main():
    df = pd.read_csv("data/output/aqhi_raw.csv")
    
    processor = AQHIProcessor()
    processed_df = processor.process(df)
    
    validator = DataValidator()
    validated_df = processed_df[processed_df.apply(validator.validate_row, axis=1)]
    
    handler = SupabaseHandler()
    handler.upsert_sensor_readings(validated_df.to_dict('records'))
    
    logger.info(f"Processed {len(validated_df)} AQHI records")

# scripts/process_purpleair.py
"""
Process raw PurpleAir → PM2.5 selection + RH correction → store
Usage: python scripts/process_purpleair.py --region AB
"""
def main(region="AB"):
    df = pd.read_json(f"data/output/purpleair_{region}_raw.json")
    
    processor = PM25Processor()
    processed_df = processor.process_batch(df)
    
    validator = DataValidator()
    validated_df = processed_df[processed_df.apply(validator.validate_row, axis=1)]
    
    handler = SupabaseHandler()
    handler.upsert_sensor_readings(validated_df.to_dict('records'))
    
    logger.info(f"Processed {len(validated_df)} PurpleAir records for {region}")

# scripts/interpolate_grid.py
"""
Generate IDW grids for each region/airshed
Usage: python scripts/interpolate_grid.py --region ACA
"""
def main(region="AB"):
    # Get latest sensor readings from Supabase
    query = SupabaseQueries()
    latest_data = query.get_latest_sensors(region)
    
    gdf = gpd.GeoDataFrame(
        latest_data,
        geometry=gpd.points_from_xy([x['longitude'] for x in latest_data], 
                                      [x['latitude'] for x in latest_data])
    )
    
    # Interpolate
    interpolator = IDWInterpolator(f"data/shapefiles/{region}.shp")
    grid = interpolator.interpolate(gdf, gdf['value'].values)
    
    # Export
    writer = GeoJSONWriter()
    writer.write_sensor_map(grid, region)
    
    logger.info(f"Generated IDW grid for {region}")

# scripts/orchestrate.py
"""
Run complete pipeline: fetch → process → interpolate
Usage: python scripts/orchestrate.py --run-aqhi --run-pa --run-idw
"""
def main():
    logger.info("Starting data pipeline orchestration...")
    
    # Fetch
    subprocess.run(["python", "scripts/fetch_aqhi.py"])
    subprocess.run(["python", "scripts/fetch_purpleair.py", "--region", "AB"])
    subprocess.run(["python", "scripts/fetch_purpleair.py", "--region", "SK"])
    
    # Process
    subprocess.run(["python", "scripts/process_aqhi.py"])
    subprocess.run(["python", "scripts/process_purpleair.py", "--region", "AB"])
    subprocess.run(["python", "scripts/process_purpleair.py", "--region", "SK"])
    
    # Interpolate
    for region in ["AB", "ACA", "PAZA", "PAS"]:
        subprocess.run(["python", "scripts/interpolate_grid.py", "--region", region])
    
    logger.info("Pipeline complete!")

# scripts/health_check.py
"""
Monitor data freshness + quality
Usage: python scripts/health_check.py
"""
def main():
    query = SupabaseQueries()
    latest = query.get_latest_sensors("AB", limit=1)
    
    if not latest:
        alert("No data in Supabase!")
        return
    
    age_hours = (datetime.now(UTC) - latest[0]['recorded_at']).total_seconds() / 3600
    
    if age_hours > 6:
        alert(f"Data stale: {age_hours:.1f} hours old")
    else:
        logger.info(f"✓ Data fresh ({age_hours:.1f}h old)")


    output_file = OUTPUT_DIR / "aqhi_raw.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Wrote to {output_file}")

if __name__ == "__main__":
    main()
