"""Alberta AQHI API client"""
import pandas as pd
from src.ingestion.base_client import BaseAPIClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AQHIClient(BaseAPIClient):
    """Client for Alberta AQHI OData API"""
    
    STATIONS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    MEASUREMENTS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"
    
    def fetch_stations(self):
        """Fetch all AQHI station metadata"""
        logger.info("Fetching AQHI station list...")
        
        params = {
            "$select": "Name,Latitude,Longitude",
            "$format": "json",
            "$top": 1000
        }
        
        response = self.get(self.STATIONS_URL, params=params)
        data = response.json()
        
        df = pd.json_normalize(data.get("value", []))
        logger.info(f"Fetched {len(df)} stations")
        return df
    
    def fetch_measurements(self, station_name: str, hours_back: int = 24):
        """Fetch measurements for a station"""
        logger.info(f"Fetching measurements for {station_name} (last {hours_back}h)...")
        
        # Safe station name for OData filter
        safe_name = station_name.replace("'", "''")
        
        params = {
            "$select": "StationName,ParameterName,ReadingDate,Value",
            "$filter": f"StationName eq '{safe_name}'",
            "$orderby": "ReadingDate desc",
            "$format": "json",
            "$top": 10000
        }
        
        response = self.get(self.MEASUREMENTS_URL, params=params)
        data = response.json()
        
        df = pd.DataFrame(data.get("value", []))
        logger.info(f"Fetched {len(df)} measurements")
        return df
