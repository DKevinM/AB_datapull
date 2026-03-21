#!/usr/bin/env python3
"""
Generate IDW interpolated grid for a region/airshed.

Usage:
    python scripts/interpolate_grid.py --region AB
    python scripts/interpolate_grid.py --region ACA
    python scripts/interpolate_grid.py --region PAZA
    python scripts/interpolate_grid.py --region PAS
    python scripts/interpolate_grid.py --region SK
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd

from config.settings import OUTPUT_DIR, SHAPEFILES
from src.utils.logger import setup_logger
from src.processing.aqhi_processor import AQHIProcessor
from src.processing.geospatial import IDWInterpolator
from src.storage.geojson_writer import GeoJSONWriter

logger = setup_logger(__name__)

# Mapping of region → input CSV and value column
REGION_PARAMS = {
    "AB": {"input_csv": "aqhi_live.csv", "value_col": "AQHI", "output_prefix": "aqhi_AB"},
    "ACA": {"input_csv": "aqhi_live.csv", "value_col": "AQHI", "output_prefix": "aqhi_ACA"},
    "PAZA": {"input_csv": "aqhi_live.csv", "value_col": "AQHI", "output_prefix": "aqhi_PAZA"},
    "PAS": {"input_csv": "aqhi_live.csv", "value_col": "AQHI", "output_prefix": "aqhi_PAS"},
    "SK": {"input_csv": "purpleair_sk_live.json", "value_col": "pm_corrected", "output_prefix": "pm25_SK"},
}


def main():
    parser = argparse.ArgumentParser(description="Generate IDW interpolation grid for a region.")
    parser.add_argument(
        "--region",
        choices=list(REGION_PARAMS.keys()),
        required=True,
        help="Region code: AB, ACA, PAZA, PAS, or SK",
    )
    args = parser.parse_args()
    region = args.region

    config = REGION_PARAMS[region]
    shapefile = SHAPEFILES.get(region)
    if not shapefile or not Path(shapefile).exists():
        logger.error(f"Shapefile not found for region={region}: {shapefile}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Generating IDW grid for region={region}")

    # Load processed data
    input_path = OUTPUT_DIR / config["input_csv"]
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}. Run process_aqhi.py or process_purpleair.py first.")
        sys.exit(1)

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path)

    value_col = config["value_col"]
    lat_col = "Latitude" if "Latitude" in df.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in df.columns else "longitude"

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[value_col, lat_col, lon_col])

    if df.empty:
        logger.error(f"No valid rows for interpolation (value_col={value_col}).")
        sys.exit(1)

    logger.info(f"Using {len(df)} points for interpolation")

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    values = df[value_col].to_numpy(dtype=float)

    # Interpolate
    interpolator = IDWInterpolator(shapefile_path=shapefile)
    grid_gdf = interpolator.interpolate(
        points_gdf=gdf,
        values=values,
        value_col=value_col,
        timestamp_col="NearestReading",
    )

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = config["output_prefix"]

    csv_path = OUTPUT_DIR / f"{prefix}_idw.csv"
    grid_gdf[["lon", "lat", value_col, "NearestReading"]].to_csv(csv_path, index=False)
    logger.info(f"Saved IDW CSV to {csv_path}")

    writer = GeoJSONWriter(output_dir=OUTPUT_DIR)
    writer.write_geodataframe(grid_gdf, f"{prefix}_grid.geojson")

    logger.info(f"IDW grid complete: {len(grid_gdf)} points")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
