"""IDW (Inverse Distance Weighting) interpolation for regional grids"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path

from src.utils.logger import setup_logger
from config.settings import IDW_GRID_STEP_DEGREES, IDW_K_NEIGHBORS

logger = setup_logger(__name__)


class IDWInterpolator:
    """Generate IDW interpolated grids clipped to a regional boundary shapefile."""

    def __init__(self, shapefile_path: str | Path, grid_step: float = IDW_GRID_STEP_DEGREES):
        self.boundary = gpd.read_file(str(shapefile_path))
        if self.boundary.crs is None or self.boundary.crs.to_epsg() != 4326:
            self.boundary = self.boundary.to_crs(epsg=4326)
        self.grid_step = grid_step
        logger.info(f"Loaded boundary from {shapefile_path} ({len(self.boundary)} features)")

    def build_grid(self) -> gpd.GeoDataFrame:
        """Create a regular lat/lon grid clipped to the boundary."""
        union = self.boundary.unary_union
        xmin, ymin, xmax, ymax = self.boundary.total_bounds
        gx, gy = np.meshgrid(
            np.arange(xmin, xmax + self.grid_step, self.grid_step),
            np.arange(ymin, ymax + self.grid_step, self.grid_step),
        )
        grid_df = pd.DataFrame({"lon": gx.ravel(), "lat": gy.ravel()})
        grid_gdf = gpd.GeoDataFrame(
            grid_df,
            geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat),
            crs="EPSG:4326",
        )
        grid_gdf = grid_gdf[grid_gdf.geometry.within(union)].reset_index(drop=True)
        logger.info(f"Grid has {len(grid_gdf)} points at step={self.grid_step}°")
        return grid_gdf

    def interpolate(
        self,
        points_gdf: gpd.GeoDataFrame,
        values: np.ndarray,
        value_col: str = "value",
        timestamp_col: str = None,
    ) -> gpd.GeoDataFrame:
        """
        Perform IDW on measurement points and return interpolated grid.

        Args:
            points_gdf: GeoDataFrame with point geometries (EPSG:4326)
            values:     1-D array of values aligned with points_gdf
            value_col:  Name for the output value column in the grid
            timestamp_col: Optional name for nearest-station timestamp column
        """
        if points_gdf.crs is None or points_gdf.crs.to_epsg() != 4326:
            points_gdf = points_gdf.to_crs(epsg=4326)

        # Clip measurement stations to boundary
        union = self.boundary.unary_union
        mask = points_gdf.geometry.within(union) | points_gdf.geometry.touches(union)
        points_gdf = points_gdf[mask].reset_index(drop=True)
        values = values[mask.values]

        if len(points_gdf) == 0:
            raise ValueError("No measurement points lie within the regional boundary.")

        grid_gdf = self.build_grid()
        if grid_gdf.empty:
            raise ValueError("Interpolation grid is empty after clipping to boundary.")

        xy_pts = np.column_stack([points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()])
        xy_grid = np.column_stack([grid_gdf.geometry.x.to_numpy(), grid_gdf.geometry.y.to_numpy()])

        n = len(xy_pts)
        if n == 1:
            grid_gdf[value_col] = float(values[0])
            if timestamp_col:
                ts = points_gdf.get("ReadingDate", pd.Series([""] * len(points_gdf)))
                grid_gdf[timestamp_col] = str(ts.iloc[0])
        else:
            k = min(IDW_K_NEIGHBORS, n)
            tree = cKDTree(xy_pts)
            dists, idxs = tree.query(xy_grid, k=k, p=2)
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]

            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / (dists ** 2)

            zero_mask = dists == 0
            if zero_mask.any():
                for r in np.where(zero_mask.any(axis=1))[0]:
                    j = np.where(zero_mask[r])[0][0]
                    weights[r, :] = 0.0
                    weights[r, j] = 1.0

            wsum = weights.sum(axis=1)
            wsum[wsum == 0] = np.nan
            grid_gdf[value_col] = np.nansum(weights * values[idxs], axis=1) / wsum

            if timestamp_col:
                ts_arr = points_gdf.get("ReadingDate", pd.Series([""] * len(points_gdf))).astype(str).to_numpy()
                grid_gdf[timestamp_col] = ts_arr[idxs[:, 0]]

        logger.info(f"IDW complete: {len(grid_gdf)} grid points, {len(points_gdf)} stations used")
        return grid_gdf
