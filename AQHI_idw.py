import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
import folium
import branca.colormap as bcm
from shapely.geometry import LineString
from matplotlib import pyplot as plt

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
grid_gdf = grid_gdf[grid_gdf.geometry.within(airshed.unary_union)]

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

# Optionally save GeoJSON
grid_gdf.to_file("data/AQHI_grid.geojson", driver="GeoJSON")




contour_gdf = gpd.read_file("data/AQHI_grid.geojson")

center_lat = latest_df["lat"].mean()
center_lon = latest_df["lon"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

# 4a) Add contours—color by “AQHI_IDW”
zmin = contour_gdf["AQHI_IDW"].min()
zmax = contour_gdf["AQHI_IDW"].max()
pal = bcm.linear.YlOrRd_09.scale(zmin, zmax)
pal.caption = "AQHI IDW Contours"

folium.GeoJson(
    contour_gdf,
    style_function=lambda feature: {
        "color": pal(feature["properties"]["AQHI_IDW"]),
        "weight": 2,
        "opacity": 0.7
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["AQHI_IDW"],
        aliases=["AQHI Level"],
        localize=True,
        labels=True,
        sticky=False
    )
).add_to(m)
pal.add_to(m)

# 4b) Add station points exactly as before
for idx, row in latest_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.8,
        popup=folium.Popup(
            f"{row['StationName']}<br>"
            f"AQHI: {row['Value']}<br>"
            f"Time: {row['NearestReading']}",
            parse_html=True
        )
    ).add_to(m)

# 5) Save:
m.save("AQHI_contour_with_stations.html")
