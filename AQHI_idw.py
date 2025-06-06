import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import branca.colormap as bcm
from shapely.geometry import Polygon, Point, MultiPolygon
from scipy.spatial import cKDTree  
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


# Load station data
df = pd.read_csv("data/last6h.csv")
df = df[df["ParameterName"].isna() | (df["ParameterName"] == "")]
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])

# Get latest reading per station
latest_df = df.sort_values("ReadingDate").groupby("StationName").tail(1)
# Drop rows with missing info
latest_df = latest_df.dropna(subset=["Value", "Latitude", "Longitude"])

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(latest_df, geometry=gpd.points_from_xy(latest_df.Longitude, latest_df.Latitude), crs="EPSG:4326")
# gdf = gdf.to_crs("EPSG:3401")  # Alberta projection

# Load airshed boundary
airshed = gpd.read_file("data/Alberta.shp").to_crs(gdf.crs)

# Create grid
xmin, ymin, xmax, ymax = airshed.total_bounds
grid_x, grid_y = np.meshgrid(
    np.arange(xmin, xmax, 0.05),
    np.arange(ymin, ymax, 0.05)
)
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
grid_df = pd.DataFrame(grid_points, columns=["lon", "lat"])
grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat), crs=gdf.crs)
grid_gdf = grid_gdf[grid_gdf.geometry.within(airshed.union_all())]

# IDW interpolation
xy = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
values = gdf["Value"].astype(float).values
timestamps = gdf["ReadingDate"].astype(str).values
xi = np.array(list(zip(grid_gdf.geometry.x, grid_gdf.geometry.y)))

tree = cKDTree(xy)
dists, idxs = tree.query(xi, k=6, p=2)
weights = 1 / np.power(dists, 2)
weights[np.isinf(weights)] = 0
z = np.sum(weights * values[idxs], axis=1) / np.sum(weights, axis=1)
nearest_ts = timestamps[idxs[:, 0]]

grid_gdf["AQHI_IDW"] = z
grid_gdf["NearestReading"] = nearest_ts

# Save as CSV for Shiny
grid_gdf[["lon", "lat", "AQHI_IDW", "NearestReading"]].to_csv("data/AQHI_idw.csv", index=False)

# Pivot directly without reload
pivot = grid_gdf.pivot(index="lat", columns="lon", values="AQHI_IDW")


# Create pivot table (reshape to grid)
pivot = df.pivot(index="lat", columns="lon", values="AQHI_IDW")
X = pivot.columns.values
Y = pivot.index.values
Z = pivot.values

# AQHI Levels and Color Palette
aqhi_levels = list(range(1, 12))  # AQHI 1 to 11
palette = [
    "#01cbff", "#0099cb", "#016797", "#fffe03", "#ffcb00",
    "#ff9835", "#fd6866", "#fe0002", "#cc0001", "#9a0100", "#640100"
]
cmap = mcolors.ListedColormap(palette)
norm = mcolors.BoundaryNorm(boundaries=list(range(1, 13)), ncolors=len(palette))

# Plot filled contours
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, Z, levels=aqhi_levels + [12], cmap=cmap, norm=norm)
plt.close(fig)  # prevent it from displaying

# Extract polygons from contour set
contour_polys = []
for level, collection in zip(cs.levels, cs.collections):
    for path in collection.get_paths():
        for poly_coords in path.to_polygons():
            if len(poly_coords) < 3:
                continue
            poly = Polygon(poly_coords)
            if poly.is_valid:
                contour_polys.append({"geometry": poly, "AQHI": int(level)})

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(contour_polys, crs="EPSG:4326")

# Load airshed boundary and ensure CRS matches
airshed = gpd.read_file("data/Alberta.shp").to_crs(gdf.crs)

# Clip contours to Alberta boundary
airshed_union = unary_union(airshed.geometry)
gdf_clipped = gdf[gdf.geometry.intersects(airshed_union)].copy()
gdf_clipped["geometry"] = gdf_clipped.geometry.intersection(airshed_union)

# Save to GeoJSON for Leaflet
gdf_clipped.to_file("data/aqhi_map.geojson", driver="GeoJSON")

