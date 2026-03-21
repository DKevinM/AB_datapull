"""Central configuration management"""
import os
from pathlib import Path
from enum import Enum

class Environment(str, Enum):
    DEV = "development"
    PROD = "production"

ENV = Environment(os.getenv("ENVIRONMENT", "development"))
DEBUG = ENV == Environment.DEV

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHAPEFILES_DIR = DATA_DIR / "shapefiles"
SENSOR_LISTS_DIR = DATA_DIR / "sensor_lists"
OUTPUT_DIR = DATA_DIR / "output"

# API Credentials
SUPABASE_URL = os.getenv("SUPABASE_DB_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
PURPLEAIR_API_KEY = os.getenv("PURPLEAIR_API_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials. Set SUPABASE_DB_URL and SUPABASE_SERVICE_KEY")

# API Timeouts & Retries
REQUEST_TIMEOUT = 45
MAX_RETRIES = 5
RETRY_BACKOFF = 0.8

# Data Processing
PM25_OUTLIER_THRESHOLD = 2000
OZONE_OUTLIER_THRESHOLD = 150
AQHI_LOOKBACK_HOURS = int(os.getenv("AQHI_LOOKBACK_HOURS", "24"))
MAX_DATA_AGE_HOURS = 6

# Interpolation
IDW_GRID_STEP_DEGREES = 0.05
IDW_K_NEIGHBORS = 6

# Timezone
TIMEZONE = "America/Edmonton"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Regions
REGIONS = ["AB", "SK"]
