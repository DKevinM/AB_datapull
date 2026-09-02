#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from zoneinfo import ZoneInfo

AB_TZ = ZoneInfo("America/Edmonton")

# ---------- Helpers ----------
def health_canada_aqhi(no2_ppb, o3_ppb, pm25_ugm3):
    return (1000.0 / 10.4) * (
        np.exp(0.000871 * no2_ppb) +
        np.exp(0.000537 * o3_ppb) +
        np.exp(0.000487 * pm25_ugm3) - 3.0
    )

def maybe_ppm_to_ppb(x):
    if pd.isna(x):
        return x
    return x * 1000.0 if x <= 1.5 else x

# ---------- Load CSV ----------
df = pd.read_csv("data/last6h.csv")

required = {"Value","StationName","ParameterName","ReadingDate","Latitude","Longitude"}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing columns in data/last6h.csv: {sorted(missing)}")

# Parse datetimes safely; convert to AB time
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce", utc=True).dt.tz_convert(AB_TZ)

# Basic numeric coercions
for c in ["Value","Latitude","Longitude"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Path A: use unlabeled AQHI rows (blank ParameterName) ----------
param_raw = df["ParameterName"]  # keep raw (with NaN) for blank detection
is_blank = param_raw.isna() | param_raw.astype(str).str.strip().eq("")

aqhi_unlabeled = (
    df.loc[is_blank, ["StationName","Latitude","Longitude","ReadingDate","Value"]]
      .dropna(subset=["StationName","Latitude","Longitude","ReadingDate","Value"])
      .copy()
)

if not aqhi_unlabeled.empty:
    # latest per station
    aqhi_unlabeled = (aqhi_unlabeled
                      .sort_values("ReadingDate")
                      .groupby("StationName", as_index=False)
                      .tail(1)
                      .rename(columns={"Value": "AQHI"}))
    source_mode = "unlabeled-AQHI"
else:
    # ---------- Path B: compute AQHI from components ----------
    keep_params = {
        "Ozone": "O3",
        "Nitrogen Dioxide": "NO2",
        "Fine Particulate Matter": "PM25",
    }
    comp = df[df["ParameterName"].isin(keep_params.keys())].copy()
    if comp.empty:
        raise ValueError("No unlabeled AQHI rows and no O3/NO2/PM2.5 rows to compute AQHI.")

    comp["param"] = comp["ParameterName"].map(keep_params)
    comp["hour"] = comp["ReadingDate"].dt.floor("H")

    hourly = (comp.groupby(["StationName","Latitude","Longitude","param","hour"], as_index=False)["Value"]
                   .mean())

    wide = hourly.pivot_table(
        index=["StationName","Latitude","Longitude","hour"],
        columns="param", values="Value", aggfunc="mean"
    ).reset_index()

    for gas in ["O3","NO2"]:
        if gas in wide.columns:
            wide[gas] = wide[gas].apply(maybe_ppm_to_ppb)
        else:
            wide[gas] = np.nan
    if "PM25" not in wide.columns:
        wide["PM25"] = np.nan

    wide = wide.sort_values(["StationName","hour"]).copy()
    wide["O3_3h"]   = wide.groupby("StationName")["O3"].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    wide["NO2_3h"]  = wide.groupby("StationName")["NO2"].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    wide["PM25_3h"] = wide.groupby("StationName")["PM25"].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)

    ok = wide[["O3_3h","NO2_3h","PM25_3h"]].notna().all(axis=1)
    wide = wide.loc[ok].copy()
    if wide.empty:
        raise ValueError("No station has ≥3 hourly means for O3/NO2/PM2.5; cannot compute AQHI.")

    wide["AQHI"] = health_canada_aqhi(wide["NO2_3h"], wide["O3_3h"], wide["PM25_3h"])

    # latest per station
    idx = wide.groupby("StationName")["hour"].idxmax()
    aqhi_unlabeled = wide.loc[idx, ["StationName","Latitude","Longitude","hour","AQHI"]].rename(columns={"hour":"ReadingDate"})
    source_mode = "computed-from-components"

if aqhi_unlabeled.empty:
    raise ValueError("No AQHI rows available after processing.")

# ---------- Geo + boundary ----------
gdf = gpd.GeoDataFrame(
    aqhi_unlabeled,
    geometry=gpd.points_from_xy(aqhi_unlabeled["Longitude"], aqhi_unlabeled["Latitude"]),
    crs="EPSG:4326"
)

# Clip to Alberta boundary
airshed = gpd.read_file("data/Alberta.shp")
if airshed.crs != gdf.crs:
    airshed = airshed.to_crs(gdf.crs)
union = airshed.unary_union

gdf = gdf[gdf.geometry.within(union) | gdf.geometry.touches(union)].reset_index(drop=True)
if gdf.empty:
    raise ValueError("All AQHI stations fell outside Alberta.shp after clipping (CRS/bounds issue?).")

# ---------- Build grid ----------
xmin, ymin, xmax, ymax = airshed.total_bounds
step = 0.05  # degrees (~5 km)
gx, gy = np.meshgrid(np.arange(xmin, xmax + step, step),
                     np.arange(ymin, ymax + step, step))
grid_df = pd.DataFrame({"lon": gx.ravel(), "lat": gy.ravel()})
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat), crs=gdf.crs)
grid_gdf = grid_gdf[grid_gdf.geometry.within(union)].reset_index(drop=True)
if grid_gdf.empty:
    raise ValueError("Interpolation grid empty after clipping to Alberta boundary.")

# ---------- IDW ----------
xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
vals = gdf["AQHI"].to_numpy(dtype=float)
ts   = pd.to_datetime(gdf["ReadingDate"]).astype(str).to_numpy()

n = len(xy)
if n == 1:
    grid_gdf["AQHI_IDW"] = float(vals[0])
    grid_gdf["NearestReading"] = ts[0]
else:
    tree = cKDTree(xy)
    k = min(6, n)
    dists, idxs = tree.query(
        np.column_stack([grid_gdf.geometry.x.to_numpy(), grid_gdf.geometry.y.to_numpy()]),
        k=k, p=2
    )
    if k == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / (dists ** 2)

    zero = (dists == 0)
    if zero.any():
        rows = np.where(zero.any(axis=1))[0]
        for r in rows:
            j = np.where(zero[r])[0][0]
            w[r, :] = 0.0
            w[r, j] = 1.0

    wsum = w.sum(axis=1)
    wsum[wsum == 0] = np.nan
    z = np.nansum(w * vals[idxs], axis=1) / wsum
    nearest_ts = ts[idxs[:, 0]]

    grid_gdf["AQHI_IDW"] = z
    grid_gdf["NearestReading"] = nearest_ts

# ---------- Save ----------
grid_gdf[["lon","lat","AQHI_IDW","NearestReading"]].to_csv("data/AQHI_idw.csv", index=False)
grid_gdf.to_file("data/aqhi_map.geojson", driver="GeoJSON")
print(f"[AQHI_idw] mode={source_mode}, stations_used={len(gdf)}, grid_pts={len(grid_gdf)}")
