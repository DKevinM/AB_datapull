# storage/supabase_handler.py

"""Supabase database operations"""
from supabase import create_client
from src.utils.logger import setup_logger
from config.settings import SUPABASE_URL, SUPABASE_KEY

logger = setup_logger(__name__)

class SupabaseHandler:
    """Handle Supabase database operations"""
    
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def upsert_sensor_readings(self, records: list[dict], batch_size=5000):
        """Batch upsert sensor readings"""
        logger.info(f"Upserting {len(records)} records...")
        
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            try:
                response = self.client.table("sensor_readings").upsert(chunk).execute()
                logger.info(f"Upserted batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Upsert failed for batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info("All records upserted successfully")



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
