"""Write processed data to CSV files"""
from pathlib import Path
import pandas as pd

from src.utils.logger import setup_logger
from config.settings import OUTPUT_DIR

logger = setup_logger(__name__)


class CSVWriter:
    """Export sensor data to CSV."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame, filename: str) -> Path:
        """Write DataFrame to a CSV file in the output directory."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Wrote {len(df)} rows to {path}")
        return path
