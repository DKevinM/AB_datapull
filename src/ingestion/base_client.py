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
