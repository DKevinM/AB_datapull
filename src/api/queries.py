# api/queries.py

"""Supabase read queries"""
from src.utils.logger import setup_logger
from config.settings import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

logger = setup_logger(__name__)

class SupabaseQueries:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def get_latest_sensors(self, region: str, limit=500):
        """Get latest reading per sensor"""
        response = self.client.table("sensor_readings")\
            .select("*")\
            .eq("province", region)\
            .order("recorded_at", desc=True)\
            .limit(limit)\
            .execute()
        return response.data
        


class SupabaseQueries:
    """Read-only queries for frontend"""
    
    def get_latest_sensors(self, region: str, limit=500):
        """Latest reading per sensor"""
        response = self.client.table("sensor_readings")\
            .select("*")\
            .eq("province", region)\
            .order("recorded_at", desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    
    def get_interpolated_grid(self, region: str):
        """Get IDW grid for a region"""
        pass
    
    def get_time_series(self, sensor_index: int, hours_back: int = 24):
        """Historical data for a sensor"""
        pass

# api/serializers.py
class ResponseSerializer:
    """Format data for web consumption"""
    
    def to_geojson(self, records: list[dict]) -> dict:
        """Convert to GeoJSON FeatureCollection"""
        pass
    
    def to_json(self, records: list[dict]) -> str:
        """JSON for web API"""
        pass
