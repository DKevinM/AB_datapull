import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import cKDTree

# Load station data
df = pd.read_csv("data/last6h.csv")

# --- Robust AQHI filter: allow 'AQHI' name OR legacy blank/NaN ---
param = df.get("ParameterName")
is_aqhi_named = param.astype(str).str.contains("AQHI", case=False, na=False)
is_legacy_blank = param.isna() | (param == "")
df = df[is_aqhi_named | is_legacy_blank].copy()

# Types
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Latest reading per station (after filtering to AQHI rows)
latest_df = (
    df.dropna(subset=["ReadingDate"])
      .sort_values("ReadingDate")
      .groupby("StationName", as_index=False)
      .tail(1)
)

# Drop rows with missing info
latest_df = latest_df.dropna(subset=["Value", "Latitude", "Longitude"])

# Guard: need at least 2 stations for reasonable IDW
if len(latest_df) < 1:
    raise ValueError("No valid AQHI station rows after filtering. Check 'ParameterName' values in last6h.csv.")
elif len(latest_df) == 1:
    # You can still produce an output by nearest-neighbor fill later if desired
    pass

# GeoDataFrame
gdf = gpd.GeoDataFrame(
    latest_df,
    geometry=gpd.points_from_xy(latest_df.Longitude, latest_df.Latitude),
    crs="EPSG:4326"
)

# Load airshed boundary
airshed = gpd.read_file("data/Alberta.shp").to_crs(gdf.crs)

# Create grid (0.05° ~ ~5 km; adjust if needed)
xmin, ymin, xmax, ymax = airshed.total_bounds
grid_x, grid_y = np.meshgrid(
    np.arange(xmin, xmax, 0.05),
    np.arange(ymin, ymax, 0.05)
)
grid_df = pd.DataFrame({"lon": grid_x.ravel(), "lat": grid_y.ravel()})
grid_gdf = gpd.GeoDataFrame(
    grid_df,
    geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat),
    crs=gdf.crs
)
grid_gdf = grid_gdf[grid_gdf.geometry.within(airshed.union_all())]

# --- IDW prep (safe shapes) ---
xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])  # (n,2)
values = gdf["Value"].to_numpy(dtype=float)                                   # (n,)
timestamps = gdf["ReadingDate"].astype(str).to_numpy()                        # (n,)
xi = np.column_stack([grid_gdf.geometry.x.to_numpy(), grid_gdf.geometry.y.to_numpy()])

# KDTree with safe k
tree = cKDTree(xy)
k_req = 6
k_eff = min(k_req, len(xy))  # don’t ask for more neighbors than stations

dists, idxs = tree.query(xi, k=k_eff, p=2)

# Ensure 2D shapes even when k_eff==1
if k_eff == 1:
    dists = dists[:, None]
    idxs = idxs[:, None]

# Inverse-distance weights with zero-distance handling
with np.errstate(divide="ignore"):
    weights = 1.0 / np.power(dists, 2)  # (m,k)
# If a grid point exactly matches a station (dist=0), give it full weight of that station
zero_mask = np.isinf(weights) | (dists == 0)
if zero_mask.any():
    # set weight 1.0 for the zero-distance neighbor and 0 for others in that row
    rows = np.where(zero_mask.any(axis=1))[0]
    weights[rows, :] = 0.0
    weights[rows, zero_mask[rows].argmax(axis=1)] = 1.0

w_sum = weights.sum(axis=1)
# Guard against rows with all-zero weights (shouldn't happen, but be safe)
w_sum[w_sum == 0] = np.nan

z = np.nansum(weights * values[idxs], axis=1) / w_sum
nearest_ts = timestamps[idxs[:, 0]]

grid_gdf["AQHI_IDW"] = z
grid_gdf["NearestReading"] = nearest_ts

# Save as CSV for Shiny
grid_gdf[["lon", "lat", "AQHI_IDW", "NearestReading"]].to_csv("data/AQHI_idw.csv", index=False)

# Save to GeoJSON for Leaflet
grid_gdf.to_file("data/aqhi_map.geojson", driver="GeoJSON")

print(f"[AQHI_idw] stations_used={len(gdf)}, k_eff={k_eff}, grid_pts={len(grid_gdf)}")
