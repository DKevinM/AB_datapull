"""PM2.5 processing: dual-channel selection & RH correction"""
import pandas as pd
from src.utils.logger import setup_logger
from config.settings import PM25_OUTLIER_THRESHOLD

logger = setup_logger(__name__)


class PM25Processor:
    """Handle PM2.5 dual-channel selection and corrections"""
    
    @staticmethod
    def get_best_pm(a, b, avg):
        """Select best PM2.5 value from dual channels A, B and their average."""
        if pd.isna(a) and not pd.isna(b) and b <= PM25_OUTLIER_THRESHOLD:
            return b
        if pd.isna(b) and not pd.isna(a) and a <= PM25_OUTLIER_THRESHOLD:
            return a
        if not pd.isna(a) and a > PM25_OUTLIER_THRESHOLD and not pd.isna(b) and b <= PM25_OUTLIER_THRESHOLD:
            return b
        if not pd.isna(b) and b > PM25_OUTLIER_THRESHOLD and not pd.isna(a) and a <= PM25_OUTLIER_THRESHOLD:
            return a
        if not pd.isna(a) and not pd.isna(b):
            diff = abs(a - b)
            if diff > 500:
                return None
            if diff > 50:
                return min(a, b)
            if not pd.isna(avg) and avg >= 0:
                return avg
        return avg
    
    @staticmethod
    def correct_pm25(pm, rh):
        """Apply RH correction to PM2.5 value."""
        if pd.isna(pm):
            return None
        if pd.isna(rh):
            rh = 50
        if rh < 30:
            return pm / (1 + 0.24 / (100 / 30 - 1))
        elif rh > 70:
            return pm / (1 + 0.24 / (100 / 70 - 1))
        else:
            return pm / (1 + 0.24 / (100 / rh - 1))
    
    def select_pm(self, row: pd.Series, channel_override: dict = None) -> tuple:
        """Choose PM2.5 value with optional channel override. Returns (value, method)."""
        sid = row.get("sensor_index")
        a = row.get("pm2.5_atm_a")
        b = row.get("pm2.5_atm_b")
        avg = row.get("pm2.5_atm")
        
        if channel_override and sid in channel_override:
            override = channel_override[sid]
            if override == "OFF":
                return None, "off"
            if override == "A":
                return a, "forced_A"
            if override == "B":
                return b, "forced_B"
        
        if not pd.isna(a) and not pd.isna(b):
            diff = abs(a - b)
            if diff > 500:
                return None, "extreme_diff"
            if diff > 50:
                return min(a, b), "min_ab"
            return avg, "avg"
        
        if pd.isna(a) and not pd.isna(b):
            return b, "b_only"
        if pd.isna(b) and not pd.isna(a):
            return a, "a_only"
        return avg, "fallback"
    
    def process_batch(self, df: pd.DataFrame, channel_override: dict = None) -> pd.DataFrame:
        """Process all rows: dual-channel selection + RH correction."""
        logger.info(f"Processing {len(df)} PM2.5 records...")
        
        result = df.copy()
        
        result[["pm_raw", "pm_method"]] = result.apply(
            lambda x: pd.Series(self.select_pm(x, channel_override)),
            axis=1,
        )
        
        result["pm_corrected"] = result.apply(
            lambda x: self.correct_pm25(x["pm_raw"], x.get("humidity")),
            axis=1,
        )
        
        logger.info(f"Processed {len(result)} records")
        return result
