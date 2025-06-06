import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import branca.colormap as bcm
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree  


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




grid_gdf = grid_gdf[grid_gdf.geometry.within(airshed.union_all())]

# 2. Hexagon creation function
def create_hexagon(center_x, center_y, size):
    """Create a hexagon polygon around a center point"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points for hexagon
    return Polygon([(center_x + size*np.cos(a), center_y + size*np.sin(a)) for a in angles])

# 3. Convert points to hexbins if needed
if all(contour_gdf.geometry.geom_type == 'Point'):
    print("Creating hexbin grid...")
    hex_size = 0.02  # Adjust this based on your coordinate system
    hexagons = []
    aqhi_values = []
    
    for idx, row in contour_gdf.iterrows():
        point = row.geometry
        hexagon = create_hexagon(point.x, point.y, hex_size)
        hexagons.append(hexagon)
        aqhi_values.append(row['AQHI_IDW'])
    
    contour_gdf = gpd.GeoDataFrame({
        'geometry': hexagons,
        'AQHI_IDW': aqhi_values
    }, crs=contour_gdf.crs)

# 4. Categorize AQHI values
def bucket_aqhi(val):
    if pd.isna(val):
        return "NA"
    x = float(val)
    return "10+" if x >= 10 else str(int(np.floor(x)))

contour_gdf["AQHI_cat"] = contour_gdf["AQHI_IDW"].apply(bucket_aqhi)
category_to_numeric = {str(i): float(i) for i in range(1, 11)}
category_to_numeric["10+"] = 11.0
contour_gdf["AQHI_num"] = contour_gdf["AQHI_cat"].map(category_to_numeric)

# 5. Create color scale
palette = [
    "#01cbff", "#0099cb", "#016797", "#fffe03", "#ffcb00",
    "#ff9835", "#fd6866", "#fe0002", "#cc0001", "#9a0100", "#640100"
]

step_col = bcm.StepColormap(
    colors=palette,
    index=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    vmin=1.0,
    vmax=11.0
)

# 6. Simplify geometries for mobile performance
contour_gdf['geometry'] = contour_gdf['geometry'].simplify(0.005, preserve_topology=True)

# 7. Create optimized map
center_lat = latest_df["Latitude"].mean()
center_lon = latest_df["Longitude"].mean()
m = folium.Map(
    location=[center_lat, center_lon], 
    zoom_start=10, 
    tiles="CartoDB positron",
    control_scale=True
)

# 8. Add choropleth
folium.GeoJson(
    contour_gdf.__geo_interface__,
    style_function=lambda feature: {
        'fillColor': step_col(feature['properties']['AQHI_num']),
        'color': 'rgba(0,0,0,0.3)',
        'weight': 0.5,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['AQHI_cat'],
        aliases=['AQHI Category'],
        style=("font-size: 12px;")
    )
).add_to(m)

# 9. Add stations
for idx, row in latest_df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color='black',
        fill=True,
        fill_color='white',
        fill_opacity=1,
        weight=1,
        popup=f"<b>{row['StationName']}</b><br>AQHI: {row['Value']}"
    ).add_to(m)

# 10. Add colorbar and save
step_col.add_to(m)
m.save("mobile_aqhi_map.html")
