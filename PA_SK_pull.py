# PA_SK_pull.py
# Pull all PurpleAir sensors in Saskatchewan and save as CSV

import os
import requests
import pandas as pd
import geopandas as gpd
import json

from shapely.geometry import Point
from supabase import create_client, Client


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
# 4) Convert to DataFrame
df = pd.DataFrame(rows, columns=fields)

# force numeric coordinates
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])

print(df[["latitude", "longitude"]].head())
print("LAT range:", df["latitude"].min(), df["latitude"].max())
print("LON range:", df["longitude"].min(), df["longitude"].max())

# 5) GeoDataFrame for spatial filtering
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

print(gdf.crs)
print(sk.crs)



# 6) Clip to Sask polygon
sk_union = sk.unary_union   # single polygon for the province

inside = gdf[gdf.geometry.intersects(sk_union)].copy()
inside["province"] = "SK"


# 7) Save to CSV for downstream use
inside.to_csv("dataSK/SK_PA_sensors.csv", index=False)

print(f"Total sensors from API: {len(gdf)}")
print(f"Sensors inside Saskatchewan: {len(inside)}")


# 8) Push sensor metadata into Supabase
supabase = create_client(
    os.getenv("SUPABASE_DB_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

payload = inside[[
    "sensor_index",
    "name",
    "latitude",
    "longitude",
    "location_type",
    "last_seen",
    "province"
]].copy()

# Optional: compute network label now or leave it null
def infer_network(name):
    n = name.upper()
    return "OTHER"

payload["network"] = payload["name"].apply(infer_network)

response = supabase.table("purpleair_sensors_meta") \
    .upsert(payload.to_dict("records"), on_conflict="sensor_index") \
    .execute()

print("Supabase response:", response)
print(f"Attempted to upsert {len(payload)} sensors.")
