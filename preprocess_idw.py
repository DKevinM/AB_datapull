import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np

# Load station data
df = pd.read_csv("data/last6h.csv")
df = df.dropna(subset=["AQHI", "Lon", "Lat", "ReadingDate"])
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])

# Get latest reading per station
latest_df = df.sort_values("ReadingDate").groupby("StationName").tail(1)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(latest_df, geometry=gpd.points_from_xy(latest_df.Lon, latest_df.Lat), crs="EPSG:4326")
gdf = gdf.to_crs("EPSG:3401")

# Load airshed boundary
airshed = gpd.read_file("data/airshed_boundary.shp").to_crs(gdf.crs)

# Create grid
xmin, ymin, xmax, ymax = airshed.total_bounds
grid_x, grid_y = np.meshgrid(
    np.arange(xmin, xmax, 1000),
    np.arange(ymin, ymax, 1000)
)
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
grid_df = pd.DataFrame(grid_points, columns=["x", "y"])
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.x, grid_df.y), crs=gdf.crs)

# Filter by airshed
grid_gdf = grid_gdf[grid_gdf.geometry.within(airshed.unary_union)]

# IDW interpolation
xy = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
values = gdf["AQHI"].values
timestamps = gdf["ReadingDate"].astype(str).values
xi = np.array(list(zip(grid_gdf.geometry.x, grid_gdf.geometry.y)))

tree = cKDTree(xy)
dists, idxs = tree.query(xi, k=6, p=2)

weights = 1 / np.power(dists, 2)
weights[np.isinf(weights)] = 0
z = np.sum(weights * values[idxs], axis=1) / np.sum(weights, axis=1)

# Optional: assign timestamp from *nearest* station (idxs[:, 0])
nearest_ts = timestamps[idxs[:, 0]]

grid_gdf["AQHI_IDW"] = z
grid_gdf["NearestReading"] = nearest_ts

# Save
grid_gdf.to_file("data/aqhi_grid.geojson", driver="GeoJSON")
grid_gdf[["x", "y", "AQHI_IDW", "NearestReading"]].to_csv("data/aqhi_idw.csv", index=False)
