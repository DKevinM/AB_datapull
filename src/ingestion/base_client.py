# ingestion/base_client.py
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
