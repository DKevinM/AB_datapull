# processing/pm25_processor.py
class PM25Processor:
    """Dual-channel PM2.5 selection + RH correction"""
    
    def select_pm(self, row: dict) -> tuple[float | None, str]:
        """
        Choose PM2.5 value: channel A, B, average, or reject
        Returns: (pm_value, method_used)
        """
        pass
    
    def apply_rh_correction(self, pm: float, rh: float) -> float:
        """RH-adjusted PM2.5"""
        pass
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all rows with validation"""
        pass

# processing/aqhi_processor.py
class AQHIProcessor:
    """AQHI calculation: unlabeled vs. computed"""
    
    def get_unlabeled_aqhi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract pre-labeled AQHI rows"""
        pass
    
    def compute_aqhi_from_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute from O3/NO2/PM2.5 if unlabeled missing"""
        pass
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point"""
        pass

# processing/geospatial.py
class IDWInterpolator:
    """Inverse Distance Weighting for grid generation"""
    
    def __init__(self, shapefile_path: str, grid_step: float = 0.05):
        self.boundary = gpd.read_file(shapefile_path)
    
    def interpolate(self, points: gpd.GeoDataFrame, values: np.ndarray) -> gpd.GeoDataFrame:
        """
        Generate interpolated grid using cKDTree + IDW
        Returns: GeoDataFrame with interpolated values
        """
        pass
    
    def clip_to_region(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Keep only points within boundary"""
        pass

# processing/validators.py
class DataValidator:
    """Outlier detection, quality checks"""
    
    OUTLIER_THRESHOLDS = {
        'PM25': 2000,
        'Ozone': 150,
        'NO2': 500,
    }
    
    def validate_row(self, row: dict) -> bool:
        """Check against thresholds"""
        pass
    
    def flag_stale_data(self, timestamp: datetime, max_age_hours: int = 6) -> bool:
        pass
