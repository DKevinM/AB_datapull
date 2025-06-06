import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import branca.colormap as bcm
from shapely.geometry import Polygon, Point, MultiPolygon
from scipy.spatial import cKDTree  
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

# Optionally save GeoJSON
grid_gdf.to_file("data/AQHI_grid.geojson", driver="GeoJSON")





# 1a) Extract arrays of (lon, lat) and their AQHI values
points = np.vstack((grid_gdf["lon"].values, grid_gdf["lat"].values)).T
values = grid_gdf["AQHI_IDW"].values

# 1b) Decide on a regular mesh resolution
nx = 200  # number of columns
ny = 220  # number of rows


# 1c) Build 1D arrays of coordinates from min→max
xi = np.linspace(grid_gdf["lon"].min(), grid_gdf["lon"].max(), nx)
yi = np.linspace(grid_gdf["lat"].min(), grid_gdf["lat"].max(), ny)

# 1d) Create 2D meshgrid (XI, YI) for interpolation
XI, YI = np.meshgrid(xi, yi)

# 1e) Interpolate using ‘linear’ (or 'nearest', 'cubic' if you prefer)
ZI = griddata(points, values, (XI, YI), method="linear")


# 2a) Choose break‐points. Here: bands [0–1,1–2,…,9–10,10+].
levels = list(range(0, 11))  # [0,1,2,...,10]
# extend='max' ensures ≥10 falls into the last band

fig, ax = plt.subplots()
CF = ax.contourf(
    XI, YI, ZI,
    levels=levels,
    extend="max",          # bottom: below 0 if needed; top: ≥10
    cmap="YlOrRd"
)
plt.close(fig)





records = []
for idx, level_value in enumerate(CF.levels):
    # CF.collections[idx] contains all patches where Z is between 
    # levels[idx] and levels[idx+1], except for the top band (extend="max").
    collection = CF.collections[idx]
    for path in collection.get_paths():
        # Each "path" is essentially a closed polygon outline in screen coords,
        # but `.vertices` returns an Nx2 array of (x,y) in data coordinates.
        coords = path.vertices
        if coords.shape[0] < 3:
            continue  # skip if fewer than 3 points (not a polygon)
        poly = Polygon(coords)
        # Label the category
        if idx < len(levels)-1:
            cat = f"{levels[idx]}–{levels[idx+1]}"
        else:
            cat = "10+"
        records.append({
            "geometry": poly,
            "AQHI_min": levels[idx],
            "AQHI_max": float("inf") if idx == len(levels)-1 else levels[idx+1],
            "AQHI_cat": cat
        })

contour_poly_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")




contour_cat = contour_poly_gdf.dissolve(by="AQHI_cat", as_index=False)
# Re‐create a numeric “bucket” for each cat so we can map to colors:
def cat_to_num(cat):
    return 11.0 if cat == "10+" else float(cat.split("–")[0])

contour_cat["bucket"] = contour_cat["AQHI_cat"].map(cat_to_num)

# (Optional) Simplify geometry to drop tiny vertices 
contour_cat["geometry"] = contour_cat["geometry"].simplify(
    tolerance=0.001, preserve_topology=True
)

palette = [
    "#01cbff", "#0099cb", "#016797", "#fffe03", "#ffcb00",
    "#ff9835", "#fd6866", "#fe0002", "#cc0001", "#9a0100", "#640100"
]
# Build a StepColormap for values 0–1→1, 1–2→2, …, 9–10→10, 10+→11
# We'll attach these numeric "bucket codes" into the GeoDataFrame next.

# 5a) Assign a numeric “bucket code” to each category exactly as above:
def cat_to_num(cat):
    if cat == "10+":
        return 11.0
    # else cat is a string "X–Y"; split on “–” and take the lower bound as float
    lower = float(cat.split("–")[0])
    return lower

contour_cat["bucket"] = contour_cat["AQHI_cat"].map(cat_to_num)

step_col = bcm.StepColormap(
    colors=palette,
    index=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    vmin=0.0,
    vmax=11.0,
)
# Note: we included a “0.0” starting index so that “0–1” maps to the first color.
step_col.caption = "AQHI Bands (0–1, 1–2, …, 9–10, 10+)"


# Center the map
center_lat = contour_cat.geometry.centroid.y.mean()
center_lon = contour_cat.geometry.centroid.x.mean()
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="CartoDB positron"
)


# 6b) Draw each category’s polygon(s) via GeoJson + style_function
folium.GeoJson(
    data=contour_cat.to_json(),
    style_function=lambda feature: {
        "fillColor": step_col(feature["properties"]["bucket"]),
        "color":     "black",
        "weight":    1,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["AQHI_cat"],
        aliases=["Band"],
        localize=True,
        labels=True
    )
).add_to(m)

# 6c) Add the discrete legend
step_col.add_to(m)



# 6d) Overlay station points (LATEST_STATIONS must have ['Latitude','Longitude','StationName','Value','Timestamp'])
for _, row in latest_df.iterrows():
    ts = str(row["Timestamp"])
    popup_html = (
        f"<b>{row['StationName']}</b><br>"
        f"AQHI: {row['Value']}<br>"
        f"Time: {ts}"
    )
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4,
        color="black",
        fill=True,
        fill_color="white",
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, parse_html=True),
    ).add_to(m)

# 6e) Save the HTML
m.save("aqhi_map.html")
