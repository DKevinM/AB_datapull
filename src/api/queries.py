# api/queries.py
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
