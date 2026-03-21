"""Data validation: outlier detection and quality checks"""
from datetime import datetime, timezone, timedelta
import pandas as pd

from src.utils.logger import setup_logger
from config.settings import PM25_OUTLIER_THRESHOLD, OZONE_OUTLIER_THRESHOLD, MAX_DATA_AGE_HOURS

logger = setup_logger(__name__)

OUTLIER_THRESHOLDS: dict[str, float] = {
    "PM25": PM25_OUTLIER_THRESHOLD,
    "Fine Particulate Matter": PM25_OUTLIER_THRESHOLD,
    "Ozone": OZONE_OUTLIER_THRESHOLD,
    "Nitrogen Dioxide": 500.0,
    "AQHI": 50.0,
}


class DataValidator:
    """Outlier detection and data-quality checks."""

    def validate_value(self, parameter: str, value: float) -> bool:
        """Return True if value is within acceptable range for the given parameter."""
        if pd.isna(value):
            return False
        threshold = OUTLIER_THRESHOLDS.get(parameter)
        if threshold is not None and value > threshold:
            logger.debug(f"Outlier rejected: {parameter}={value} > {threshold}")
            return False
        return True

    def filter_dataframe(self, df: pd.DataFrame, param_col: str = "ParameterName", value_col: str = "Value") -> pd.DataFrame:
        """Remove rows that fail outlier checks."""
        before = len(df)
        mask = df.apply(lambda r: self.validate_value(r.get(param_col, ""), r.get(value_col)), axis=1)
        filtered = df[mask].copy()
        dropped = before - len(filtered)
        if dropped:
            logger.warning(f"Dropped {dropped} outlier rows out of {before}")
        return filtered

    def is_stale(self, recorded_at: datetime, max_age_hours: int = MAX_DATA_AGE_HOURS) -> bool:
        """Return True if the timestamp is older than max_age_hours."""
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - recorded_at
        return age > timedelta(hours=max_age_hours)
