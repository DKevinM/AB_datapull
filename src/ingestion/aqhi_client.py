"""Alberta AQHI API client"""
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from src.ingestion.base_client import BaseAPIClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

AB_TZ = ZoneInfo("America/Edmonton")


class AQHIClient(BaseAPIClient):
    """Client for Alberta AQHI OData API"""
    
    STATIONS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    MEASUREMENTS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"
    
    def fetch_stations(self) -> pd.DataFrame:
        """Fetch all AQHI station metadata"""
        logger.info("Fetching AQHI station list...")
        
        params = {
            "$select": "Name,Latitude,Longitude",
            "$format": "json",
            "$top": 1000,
        }
        
        response = self.get(self.STATIONS_URL, params=params)
        data = response.json()
        
        df = pd.json_normalize(data.get("value", []))
        logger.info(f"Fetched {len(df)} stations")
        return df
    
    def fetch_measurements(self, station_name: str, hours_back: int = 24) -> pd.DataFrame:
        """Fetch measurements for a station within the last N hours"""
        now_ab = datetime.now(AB_TZ)
        start_ab = now_ab - timedelta(hours=hours_back)
        # Format as ISO offset string: 2025-11-01T21:34:20-06:00
        start_str = start_ab.isoformat(timespec="seconds")
        
        safe_name = station_name.replace("'", "''")
        params = {
            "$select": "StationName,ParameterName,ReadingDate,Value",
            "$filter": f"StationName eq '{safe_name}' AND ReadingDate gt {start_str}",
            "$orderby": "ReadingDate desc",
            "$format": "json",
            "$top": 10000,
        }
        
        response = self.get(self.MEASUREMENTS_URL, params=params)
        data = response.json()
        
        df = pd.DataFrame(data.get("value", []))
        if not df.empty:
            logger.debug(f"Fetched {len(df)} measurements for {station_name}")
        return df
