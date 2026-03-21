"""AQHI data processing: unlabeled rows vs. computed from components"""
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

AB_TZ = ZoneInfo("America/Edmonton")

# Parameters measured in ppm that need to be converted to ppb
PPM_PARAMS = {
    "Ozone",
    "Total Oxides of Nitrogen",
    "Hydrogen Sulphide",
    "Total Reduced Sulphur",
    "Sulphur Dioxide",
    "Nitric Oxide",
    "Nitrogen Dioxide",
}


def health_canada_aqhi(no2_ppb: float, o3_ppb: float, pm25_ugm3: float) -> float:
    """Health Canada AQHI formula using 3-hour averages."""
    return (1000.0 / 10.4) * (
        np.exp(0.000537 * no2_ppb)
        + np.exp(0.000871 * o3_ppb)
        + np.exp(0.000487 * pm25_ugm3)
        - 3.0
    )


class AQHIProcessor:
    """AQHI calculation: uses unlabeled rows when available, otherwise computes from components."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Returns a DataFrame with columns:
        StationName, Latitude, Longitude, ReadingDate, AQHI, source_mode
        """
        df = df.copy()
        df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce", utc=True).dt.tz_convert(AB_TZ)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df[["Latitude", "Longitude"]] = df[["Latitude", "Longitude"]].apply(
            pd.to_numeric, errors="coerce"
        )

        # Apply ppm → ppb conversion
        df.loc[df["ParameterName"].isin(PPM_PARAMS), "Value"] *= 1000

        result = self._get_unlabeled_aqhi(df)
        if not result.empty:
            result["source_mode"] = "unlabeled-AQHI"
            logger.info(f"Using unlabeled AQHI rows for {len(result)} stations")
            return result

        result = self._compute_from_components(df)
        result["source_mode"] = "computed-from-components"
        logger.info(f"Computed AQHI from components for {len(result)} stations")
        return result

    def _get_unlabeled_aqhi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract pre-labeled (blank ParameterName) AQHI rows."""
        is_blank = df["ParameterName"].isna() | df["ParameterName"].astype(str).str.strip().eq("")
        unlabeled = df.loc[is_blank].dropna(subset=["StationName", "Latitude", "Longitude", "ReadingDate", "Value"]).copy()

        if unlabeled.empty:
            return pd.DataFrame()

        latest = (
            unlabeled.sort_values("ReadingDate")
            .groupby("StationName", as_index=False)
            .last()
            .rename(columns={"Value": "AQHI"})
        )
        return latest[["StationName", "Latitude", "Longitude", "ReadingDate", "AQHI"]]

    def _compute_from_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute AQHI from O3, NO2, PM2.5 when unlabeled rows are absent."""
        keep_params = {"Ozone": "O3", "Nitrogen Dioxide": "NO2", "Fine Particulate Matter": "PM25"}
        comp = df[df["ParameterName"].isin(keep_params)].copy()
        if comp.empty:
            raise ValueError("No O3/NO2/PM2.5 rows available to compute AQHI.")

        comp["param"] = comp["ParameterName"].map(keep_params)
        comp["hour"] = comp["ReadingDate"].dt.floor("h")

        hourly = comp.groupby(
            ["StationName", "Latitude", "Longitude", "param", "hour"], as_index=False
        )["Value"].mean()

        wide = hourly.pivot_table(
            index=["StationName", "Latitude", "Longitude", "hour"],
            columns="param",
            values="Value",
            aggfunc="mean",
        ).reset_index()

        for gas in ["O3", "NO2", "PM25"]:
            if gas not in wide.columns:
                wide[gas] = np.nan

        wide = wide.sort_values(["StationName", "hour"])
        for col, roll_col in [("O3", "O3_3h"), ("NO2", "NO2_3h"), ("PM25", "PM25_3h")]:
            wide[roll_col] = (
                wide.groupby("StationName")[col]
                .rolling(3, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
            )

        ok = wide[["O3_3h", "NO2_3h", "PM25_3h"]].notna().all(axis=1)
        wide = wide.loc[ok].copy()
        if wide.empty:
            raise ValueError("No station has ≥3 hourly means for O3/NO2/PM2.5.")

        wide["AQHI"] = health_canada_aqhi(wide["NO2_3h"], wide["O3_3h"], wide["PM25_3h"])

        idx = wide.groupby("StationName")["hour"].idxmax()
        latest = wide.loc[idx, ["StationName", "Latitude", "Longitude", "hour", "AQHI"]].rename(
            columns={"hour": "ReadingDate"}
        )
        return latest
