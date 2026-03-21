# storage/supabase_handler.py
class SupabaseHandler:
    """Database operations"""
    
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def upsert_sensor_readings(self, records: list[dict], batch_size=5000):
        """Batch upsert with chunking"""
        for chunk in self._chunk(records, batch_size):
            response = self.client.table("sensor_readings").upsert(chunk).execute()
            logger.info(f"Upserted {len(chunk)} records")
    
    def upsert_interpolated_grid(self, grid_gdf: gpd.GeoDataFrame, region: str):
        """Store interpolated data"""
        pass

# storage/csv_writer.py
class CSVWriter:
    """Export to CSV"""
    
    def write_sensor_data(self, df: pd.DataFrame, region: str):
        path = f"data/output/{region}_readings.csv"
        df.to_csv(path, index=False)
        logger.info(f"Wrote {len(df)} rows to {path}")

# storage/geojson_writer.py
class GeoJSONWriter:
    """Export to GeoJSON"""
    
    def write_sensor_map(self, gdf: gpd.GeoDataFrame, region: str):
        path = f"data/output/{region}_map.geojson"
        gdf.to_file(path, driver="GeoJSON")
        logger.info(f"Wrote GeoJSON to {path}")
