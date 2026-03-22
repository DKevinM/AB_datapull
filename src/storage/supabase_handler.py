"""Supabase database operations"""
from supabase import create_client
from src.utils.logger import setup_logger
from config.settings import SUPABASE_URL, SUPABASE_KEY

logger = setup_logger(__name__)


class SupabaseHandler:
    """Handle Supabase database operations"""
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials. Set SUPABASE_DB_URL and SUPABASE_SERVICE_KEY")
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def upsert_sensor_readings(self, records: list[dict], batch_size: int = 5000) -> int:
        """Batch upsert sensor readings. Returns total records upserted."""
        if not records:
            logger.warning("No records to upsert")
            return 0
        
        logger.info(f"Upserting {len(records)} records...")
        total = 0
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            try:
                self.client.table("sensor_readings").upsert(chunk).execute()
                total += len(chunk)
                logger.info(f"Upserted batch {i // batch_size + 1} ({len(chunk)} records)")
            except Exception as e:
                logger.error(f"Upsert failed for batch {i // batch_size + 1}: {e}")
                raise
        
        logger.info(f"All {total} records upserted successfully")
        return total
    
    def upsert_aqhi_readings(self, records: list[dict]) -> int:
        """Upsert AQHI station readings."""
        if not records:
            logger.warning("No AQHI records to upsert")
            return 0
        logger.info(f"Upserting {len(records)} AQHI records...")
        self.client.table("aqhi_readings").upsert(
            records, on_conflict="station_name,province,recorded_at"
        ).execute()
        logger.info(f"Upserted {len(records)} AQHI records")
        return len(records)

    def upsert_stations(self, records: list[dict]) -> int:
        """Upsert AQHI station metadata."""
        if not records:
            return 0
        logger.info(f"Upserting {len(records)} station records...")
        self.client.table("stations").upsert(records, on_conflict="StationName").execute()
        logger.info(f"Upserted {len(records)} stations")
        return len(records)
    
    def upsert_purpleair_sensors(self, records: list[dict]) -> int:
        """Upsert PurpleAir sensor metadata."""
        if not records:
            return 0
        logger.info(f"Upserting {len(records)} PurpleAir sensor records...")
        self.client.table("purpleair_sensors_meta").upsert(
            records, on_conflict="sensor_index"
        ).execute()
        logger.info(f"Upserted {len(records)} PurpleAir sensors")
        return len(records)
