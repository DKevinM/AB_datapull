"""PurpleAir API client"""
import pandas as pd
from src.ingestion.base_client import BaseAPIClient
from src.utils.logger import setup_logger
from config.settings import PURPLEAIR_API_KEY

logger = setup_logger(__name__)


class PurpleAirClient(BaseAPIClient):
    """Client for PurpleAir API v1"""
    
    BASE_URL = "https://api.purpleair.com/v1/sensors"
    
    # Default fields to request from the API
    DEFAULT_FIELDS = "sensor_index,name,latitude,longitude,location_type,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b"

    def __init__(self):
        super().__init__()
        if not PURPLEAIR_API_KEY:
            raise ValueError("PURPLEAIR_API_KEY not set")
        self.api_key = PURPLEAIR_API_KEY
        self.session.headers.update({"X-API-Key": self.api_key})

    def fetch_sensors(self, sensor_ids: list[int]) -> pd.DataFrame:
        """Fetch sensor data for a specific list of sensor IDs."""
        logger.info(f"Fetching {len(sensor_ids)} PurpleAir sensors by ID...")
        
        sensor_id_str = ",".join(map(str, sensor_ids))
        params = {
            "fields": self.DEFAULT_FIELDS,
            "show_only": sensor_id_str,
        }
        
        response = self.get(self.BASE_URL, params=params)
        data = response.json()
        
        df = pd.DataFrame(data.get("data", []), columns=data.get("fields", []))
        logger.info(f"Fetched {len(df)} sensors")
        return df

    def fetch_sensors_bbox(
        self,
        nwlng: float,
        nwlat: float,
        selng: float,
        selat: float,
        location_type: int = 0,
    ) -> pd.DataFrame:
        """Fetch all outdoor sensors within a bounding box."""
        logger.info(f"Fetching PurpleAir sensors in bbox: NW ({nwlat},{nwlng}) SE ({selat},{selng})")
        
        params = {
            "fields": self.DEFAULT_FIELDS,
            "location_type": location_type,
            "nwlng": nwlng,
            "nwlat": nwlat,
            "selng": selng,
            "selat": selat,
        }
        
        response = self.get(self.BASE_URL, params=params)
        data = response.json()
        
        df = pd.DataFrame(data.get("data", []), columns=data.get("fields", []))
        logger.info(f"Fetched {len(df)} sensors from bounding box")
        return df
