"""Supabase read queries"""
from datetime import datetime, timezone
from src.utils.logger import setup_logger
from config.settings import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

logger = setup_logger(__name__)


class SupabaseQueries:
    """Read-only queries for frontend / health checks"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_DB_URL and SUPABASE_SERVICE_KEY must be set")
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def get_latest_sensors(self, province: str = None, limit: int = 500) -> list[dict]:
        """Get the most recent readings, optionally filtered by province."""
        query = self.client.table("sensor_readings").select("*")
        if province:
            query = query.eq("province", province)
        response = query.order("recorded_at", desc=True).limit(limit).execute()
        return response.data
    
    def get_latest_reading_timestamp(self) -> datetime | None:
        """Return the most recent recorded_at across all sensor readings."""
        response = (
            self.client.table("sensor_readings")
            .select("recorded_at")
            .order("recorded_at", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return datetime.fromisoformat(response.data[0]["recorded_at"])
        return None
    
    def get_active_sensor_count(self, hours: int = 1) -> int:
        """Count sensors that reported within the last N hours."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        response = (
            self.client.table("sensor_readings")
            .select("sensor_index", count="exact")
            .gte("recorded_at", cutoff.isoformat())
            .execute()
        )
        return response.count or 0
    
    def get_time_series(self, sensor_index: int, hours_back: int = 24) -> list[dict]:
        """Historical data for a single sensor."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        response = (
            self.client.table("sensor_readings")
            .select("*")
            .eq("sensor_index", sensor_index)
            .gte("recorded_at", cutoff.isoformat())
            .order("recorded_at")
            .execute()
        )
        return response.data
