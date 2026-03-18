# PA_AB_pull.py
# Pull all PurpleAir sensors in Alberta and save as CSV

import os
import requests
import pandas as pd
import geopandas as gpd
import json

from shapely.geometry import Point
# from supabase import create_client, Client


# 1) Load Alberta boundary (can be any provincial polygon)
sk = gpd.read_file("dataSK/SK.shp")

# Make sure it’s in WGS84 (lat/lon) for the API bbox
if sk.crs is None or sk.crs.to_epsg() != 4326:
    sk = sk.to_crs(epsg=4326)

# 2) Get bounding box [minx, miny, maxx, maxy]
minx, miny, maxx, maxy = sk.total_bounds

# 3) Call PurpleAir /v1/sensors endpoint
url = "https://api.purpleair.com/v1/sensors"
headers = {"X-API-Key": os.getenv("PURPLEAIR_API_KEY")}

params = {
    # what fields you want – add more if needed later
    "fields": "sensor_index,name,latitude,longitude,location_type,last_seen,pm2.5,pm2.5_a,pm2.5_b",

    # optional: 0 = outside only, 1 = inside only
    "location_type": 0,

    # bounding box (NW and SE corners)
    "nwlng": minx,
    "nwlat": maxy,
    "selng": maxx,
    "selat": miny,
}

resp = requests.get(url, headers=headers, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

fields = data.get("fields", [])
rows   = data.get("data", [])

if not rows:
    raise RuntimeError("No sensors returned from PurpleAir – check bbox or API key.")

# 4) Convert to DataFrame
df = pd.DataFrame(rows, columns=fields)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# drop garbage rows
df = df.dropna(subset=["latitude", "longitude"])
print(df[["latitude", "longitude"]].head())
print(gdf.crs)
print(sk.crs)

# 5) GeoDataFrame for spatial filtering
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)



# 6) Clip to Alberta polygon (so we don’t keep BC/SK border sensors)
sk_union = sk.unary_union   # single polygon for the province

inside = gdf[gdf.geometry.intersects(sk_union)].copy()



# 7) Save to CSV for downstream use
features = []

for _, row in inside.iterrows():

    features.append({
        "type": "Feature",
        "properties": {
            "sensor_index": int(row["sensor_index"]),
            "name": row["name"],
            "pm25": row.get("pm2.5"),
            "pm25_a": row.get("pm2.5_a"),
            "pm25_b": row.get("pm2.5_b"),
            "last_seen": int(row["last_seen"]),
            "location_type": int(row["location_type"])
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(row["longitude"]), float(row["latitude"])]
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

outfile = "dataSK/SK_PA_sensors.geojson"

with open(outfile, "w") as f:
    json.dump(geojson, f)

print(f"Saved GeoJSON: {outfile}  features: {len(features)}")
