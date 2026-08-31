# purpleair regional pull - once per day to update sensor list
import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load shapefile
region = gpd.read_file("data/PAS_2025.shp")

# Get bounding box
bbox = region.total_bounds  # [minx, miny, maxx, maxy]

# Pull PurpleAir metadata
url = "https://api.purpleair.com/v1/sensors"
headers = {"X-API-Key": os.getenv("PURPLEAIR_API_KEY")}

params = {
    "fields": "sensor_index,name,latitude,longitude",
    "nwlng": bbox[0],
    "nwlat": bbox[3],
    "selng": bbox[2],
    "selat": bbox[1]
}

resp = requests.get(url, headers=headers, params=params)
data = resp.json()
fields = data["fields"]
rows = data["data"]

# Convert to DataFrame
df = pd.DataFrame(rows, columns=fields)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Filter to shapefile region
inside = gdf[gdf.geometry.within(region.union_all())]

# Save to CSV
inside.to_csv("data/PAS_sensors.csv", index=False)
