import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import cKDTree

# ---------------------------
# Load station data
# ---------------------------
df = pd.read_csv("data/last6h.csv")

# ---------------------------
# Find which column contains the parameter label
# ---------------------------
param_col_candidates = [
    "ParameterName", "Parameter", "Parameter_Name", "ParameterID", "ParameterId",
    "ParameterCode", "Pollutant", "PollutantName"
]
param_col = next((c for c in param_col_candidates if c in df.columns), None)
if param_col is None:
    raise KeyError(f"No parameter-name column found. Columns present: {list(df.columns)}")

# ---------------------------
# Normalize common columns early and print quick debug info
# ---------------------------
required_cols = ["ReadingDate", "Latitude", "Longitude", "Value", "StationName", param_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Required columns missing from CSV: {missing}. Columns found: {list(df.columns)}")

# Coerce types
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Print parameter label frequency (helpful in CI logs)
labels = df[param_col].astype(str)
top_vals = labels.value_counts().head(30).to_dict()
print(f"[AQHI_idw] param_col='{param_col}' top labels (top 30): {top_vals}")

# ---------------------------
# Broad AQHI matcher (tweak to match your export)
# ---------------------------
aqhi_regex = (
    r"(?i)\bAQHI\b"
    r"|(?i)Air\s*Quality\s*Health\s*Index"
    r"|(?i)\bAQHI\s*[-_/() ]*\s*(now|current|3h|3-?hr|3hour|3 hr)?"
    r"|(?i)\bHealth\s*Index\b"
)

is_aqhi_named = labels.str.contains(aqhi_regex, regex=True, na=False)
is_legacy_blank = labels.isna() | (labels.str.strip() == "")

df_aqhi = df[is_aqhi_named | is_legacy_blank].copy()

if df_aqhi.empty:
    raise ValueError(
        "No AQHI rows after filtering. See printed 'top labels' above and update aqhi_regex "
        "or fix the CSV exporter so AQHI is present."
    )

# ---------------------------
# Latest reading per station (after AQHI filter)
# ---------------------------
latest_df = (
    df_aqhi.dropna(subset=["ReadingDate"])
          .sort_values("ReadingDate")
          .groupby("StationName", as_index=False)
          .tail(1)
)

# Drop rows missing coordinates or value
latest_df = latest_df.dropna(subset=["Value", "Latitude", "Longitude"])

if latest_df.empty:
    raise ValueError("No valid AQHI station rows after selecting latest readings and dropping missing coords/values.")

# ---------------------------
# GeoDataFrame for stations
# ---------------------------
gdf = gpd.GeoDataFrame(
    latest_df,
    geometry=gpd.points_from_xy(latest_df.Longitude, latest_df.Latitude),
    crs="EPSG:4326"
)

# ---------------------------
# Load airshed boundary and build grid
# ---------------------------
airshed = gpd.read_file("data/Alberta.shp").to_crs(gdf.crs)

xmin, ymin, xmax, ymax = airshed.total_bounds
# adjust step if you want finer/coarser grid
step = 0.05
grid_x, grid_y = np.meshgrid(
    np.arange(xmin, xmax + step, step),
    np.arange(ymin, ymax + step, step)
)
grid_df = pd.DataFrame({"lon": grid_x.ravel(), "lat": grid_y.ravel()})
grid_gdf = gpd.GeoDataFrame(
    grid_df,
    geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat),
    crs=gdf.crs
)

# restrict to airshed
airshed_union = airshed.unary_union
grid_gdf = grid_gdf[grid_gdf.geometry.within(airshed_union)].reset_index(drop=True)

if grid_gdf.empty:
    raise ValueError("Interpolation grid is empty after clipping to airshed. Check Alberta.shp CRS / bounds.")

# ---------------------------
# IDW interpolation (safe handling for 1 station)
# ---------------------------
xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])  # (n,2)
values = gdf["Value"].to_numpy(dtype=float)                                   # (n,)
timestamps = gdf["ReadingDate"].astype(str).to_numpy()                        # (n,)
xi = np.column_stack([grid_gdf.geometry.x.to_numpy(), grid_gdf.geometry.y.to_numpy()])

n_stations = len(xy)
if n_stations == 0:
    raise ValueError("No stations available for interpolation after all filtering.")
elif n_stations == 1:
    # Fill grid with single station value (nearest-neighbor fallback)
    single_val = float(values[0])
    single_ts = timestamps[0]
    grid_gdf["AQHI_IDW"] = single_val
    grid_gdf["NearestReading"] = single_ts
else:
    tree = cKDTree(xy)
    k_req = 6
    k_eff = min(k_req, n_stations)

    dists, idxs = tree.query(xi, k=k_eff, p=2)
    # ensure shapes when k_eff == 1
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # inverse-distance weighting
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / np.power(dists, 2)

    # handle exact matches (dist == 0): give full weight to that neighbor
    zero_mask = (dists == 0)
    if zero_mask.any():
        rows = np.where(zero_mask.any(axis=1))[0]
        for r in rows:
            # index of the zero-distance neighbor in that row
            j = np.where(zero_mask[r])[0][0]
            weights[r, :] = 0.0
            weights[r, j] = 1.0

    w_sum = weights.sum(axis=1)
    # avoid division by zero
    w_sum[w_sum == 0] = np.nan

    z = np.nansum(weights * values[idxs], axis=1) / w_sum
    nearest_ts = timestamps[idxs[:, 0]]

    grid_gdf["AQHI_IDW"] = z
    grid_gdf["NearestReading"] = nearest_ts

# ---------------------------
# Save outputs
# ---------------------------
grid_gdf[["lon", "lat", "AQHI_IDW", "NearestReading"]].to_csv("data/AQHI_idw.csv", index=False)
grid_gdf.to_file("data/aqhi_map.geojson", driver="GeoJSON")

print(f"[AQHI_idw] stations_used={n_stations}, grid_pts={len(grid_gdf)}")
