"""PurpleAir API client"""
import pandas as pd
from src.ingestion.base_client import BaseAPIClient
from src.utils.logger import setup_logger
from config.settings import PURPLEAIR_API_KEY

logger = setup_logger(__name__)

class PurpleAirClient(BaseAPIClient):
    """Client for PurpleAir API v1"""
    
    BASE_URL = "https://api.purpleair.com/v1/sensors"
    
    def __init__(self):
        super().__init__()
        if not PURPLEAIR_API_KEY:
            raise ValueError("PURPLEAIR_API_KEY not set")
        self.api_key = PURPLEAIR_API_KEY
    
    def fetch_sensors(self, sensor_ids: list[int]):
        """Fetch sensor data by IDs"""
        logger.info(f"Fetching {len(sensor_ids)} PurpleAir sensors...")
        
        sensor_id_str = ",".join(map(str, sensor_ids))
        headers = {"X-API-Key": self.api_key}
        params = {
            "fields": "sensor_index,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b,latitude,longitude",
            "show_only": sensor_id_str
        }
        
        response = self.get(self.BASE_URL, params=params)
        data = response.json()
        
        df = pd.DataFrame(data.get("data", []), columns=data.get("fields", []))
        logger.info(f"Fetched {len(df)} sensors")
        return df
