# ingestion/base_client.py

"""Base HTTP client with retry logic"""
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

logger = logging.getLogger(__name__)

class BaseAPIClient:
    """HTTP client with automatic retry logic"""
    
    def __init__(self, timeout=45, max_retries=5):
        self.timeout = timeout
        self.session = self._create_session(max_retries)
    
    def _create_session(self, max_retries):
        """Create requests.Session with retry strategy"""
        session = requests.Session()
        retries = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
    
    def get(self, url, params=None, timeout=None):
        """GET request with error handling"""
        timeout = timeout or self.timeout
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def close(self):
        """Clean up session"""
        self.session.close()




class BaseAPIClient:
    """HTTP client with retry logic, rate limiting, error handling"""
    def __init__(self, timeout=45, max_retries=5):
        self.session = self._create_session()
    
    def _create_session(self):
        retries = Retry(total=5, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503])
        adapter = HTTPAdapter(max_retries=retries)
        session = requests.Session()
        session.mount("https://", adapter)
        return session
    
    def get(self, url, params=None):
        """GET with logging + error handling"""
        pass

# ingestion/aqhi_client.py
class AQHIClient(BaseAPIClient):
    """Alberta AQHI OData API"""
    def fetch_stations(self) -> list[dict]:
        """Fetch all AQHI stations"""
        pass
    
    def fetch_measurements(self, station_name: str, hours_back: int) -> pd.DataFrame:
        """Fetch measurements for a station"""
        pass

# ingestion/purpleair_client.py
class PurpleAirClient(BaseAPIClient):
    """PurpleAir API v1"""
    def fetch_sensors(self, region: str, sensor_ids: list[int]) -> pd.DataFrame:
        """Fetch latest sensor data by ID"""
        pass

# ingestion/models.py
from pydantic import BaseModel

class SensorReading(BaseModel):
    sensor_index: int
    province: str
    parameter_name: str
    value: float
    recorded_at: datetime
    latitude: float
    longitude: float
