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
for col in ["NearestReading", "ReadingDate"]:
    if col in contour_gdf.columns:
        contour_gdf = contour_gdf.drop(columns=[col])
        
def bucket_aqhi(val):
    if pd.isna(val):
        return "NA"
    x = float(val)
    return "10+" if x >= 10 else str(int(np.floor(x)))

contour_gdf["AQHI_cat"] = contour_gdf["AQHI_IDW"].apply(bucket_aqhi)

category_to_numeric = {str(i): float(i) for i in range(1, 11)}
category_to_numeric["10+"] = 11.0

contour_gdf["AQHI_num"] = contour_gdf["AQHI_cat"].map(category_to_numeric)

# 3) Build a categorical palette: 1→10 and "10+" exactly match your R-style colors
palette = [
    "#01cbff",  # 1
    "#0099cb",  # 2
    "#016797",  # 3
    "#fffe03",  # 4
    "#ffcb00",  # 5
    "#ff9835",  # 6
    "#fd6866",  # 7
    "#fe0002",  # 8
    "#cc0001",  # 9
    "#9a0100",  # 10
    "#640100",  # 10+
]

step_col = bcm.StepColormap(
    colors=palette,
    index=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    vmin=1.0,
    vmax=11.0
)

if "Timestamp" in latest_df.columns and pd.api.types.is_datetime64_any_dtype(
    latest_df["Timestamp"]
):
    latest_df["Timestamp"] = latest_df["Timestamp"].astype(str)
time_str = latest_df["Timestamp"].iloc[0] if "Timestamp" in latest_df.columns else "unknown"
step_col.caption = f"AQHI Category (station time: {time_str})"




center_lat = latest_df["Latitude"].mean()
center_lon = latest_df["Longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

folium.Choropleth(
    geo_data=contour_gdf.to_json(),
    data=contour_gdf,
    columns=["AQHI_num", "AQHI_num"],
    key_on="feature.properties.AQHI_num",
    fill_color=step_col,
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=step_col.caption,
    highlight=True,
    nan_fill_color="gray",
    nan_fill_opacity=0.3,
).add_to(m)

# 4b) Add station points exactly as before
for idx, row in latest_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.8,
        popup=folium.Popup(
            f"{row['StationName']}<br>"
            f"AQHI: {row['Value']}<br>"
            f"Time: {row['ReadingDate']}",
            parse_html=True
        )
    ).add_to(m)

# 5) Save:
m.save("AQHI_contour_with_stations.html")
