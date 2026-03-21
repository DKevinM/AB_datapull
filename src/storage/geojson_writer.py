"""Write processed data to GeoJSON files"""
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd

from src.utils.logger import setup_logger
from config.settings import OUTPUT_DIR

logger = setup_logger(__name__)


class GeoJSONWriter:
    """Export sensor / grid data to GeoJSON."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_geodataframe(self, gdf: gpd.GeoDataFrame, filename: str) -> Path:
        """Write a GeoDataFrame as GeoJSON."""
        path = self.output_dir / filename
        gdf.to_file(str(path), driver="GeoJSON")
        logger.info(f"Wrote GeoJSON ({len(gdf)} features) to {path}")
        return path

    def write_sensor_map(self, df: pd.DataFrame, filename: str, lat_col: str = "latitude", lon_col: str = "longitude") -> Path:
        """Build a GeoJSON FeatureCollection from a flat DataFrame."""
        features = []
        for _, row in df.iterrows():
            lat = row.get(lat_col)
            lon = row.get(lon_col)
            if pd.isna(lat) or pd.isna(lon):
                continue
            props = {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in (lat_col, lon_col)}
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            })
        geojson = {"type": "FeatureCollection", "features": features}
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(geojson, f, allow_nan=False)
        logger.info(f"Wrote {len(features)} features to {path}")
        return path
