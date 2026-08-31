# PA_AB_pull.py
# Pull all PurpleAir sensors in Alberta and save as CSV

import os
import requests
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from supabase import create_client, Client


# 1) Load Alberta boundary (can be any provincial polygon)
ab = gpd.read_file("data/Alberta.shp")

# Make sure it’s in WGS84 (lat/lon) for the API bbox
if ab.crs is None or ab.crs.to_epsg() != 4326:
    ab = ab.to_crs(epsg=4326)

# 2) Get bounding box [minx, miny, maxx, maxy]
minx, miny, maxx, maxy = ab.total_bounds

# 3) Call PurpleAir /v1/sensors endpoint
url = "https://api.purpleair.com/v1/sensors"
headers = {"X-API-Key": os.getenv("PURPLEAIR_API_KEY")}

params = {
    # what fields you want – add more if needed later
    "fields": "sensor_index,name,latitude,longitude,location_type,last_seen",

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
df["sensor_index"] = pd.to_numeric(df["sensor_index"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["last_seen"] = pd.to_numeric(df["last_seen"], errors="coerce")

df = df.dropna(subset=["sensor_index", "latitude", "longitude"])
df["sensor_index"] = df["sensor_index"].astype("int64")

df["last_seen_utc"] = pd.to_datetime(df["last_seen"], unit="s", utc=True)


now_utc = pd.Timestamp.now(tz="UTC")
df["age_days"] = (now_utc - df["last_seen_utc"]).dt.total_seconds() / 86400
df["active_30d"] = df["age_days"] <= 30
df["active_7d"] = df["age_days"] <= 7



# 5) GeoDataFrame for spatial filtering
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

# 6) Clip to Alberta polygon (so we don’t keep BC/SK border sensors)
ab_union = ab.geometry.union_all()   # single polygon for the province

inside = gdf[gdf.geometry.intersects(ab_union)].copy()
inside["province"] = "AB"

# Drop geometry before writing CSV / pushing to Supabase
inside_no_geom = pd.DataFrame(inside.drop(columns="geometry"))

# Save only recently active sensors for downstream live PurpleAir pulls
inside_live = inside_no_geom[inside_no_geom["active_30d"] == True].copy()
inside_live.to_csv("data/AB_PA_sensors.csv", index=False)

print(f"Total sensors from API: {len(gdf)}")
print(f"Sensors inside Alberta: {len(inside)}")


# 8) Push sensor metadata into Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

payload = inside_no_geom[[
    "sensor_index",
    "name",
    "latitude",
    "longitude",
    "location_type",
    "last_seen",
    "last_seen_utc",
    "age_days",
    "active_7d",
    "active_30d",
    "province"
]].copy()

# Optional: compute network label now or leave it null
def infer_network(name):
    n = name.upper()
    if "ACA" in n or "ALBERTA CAPITAL" in n:
        return "ACA"
    if "WCAS" in n or "WCA" in n:
        return "WCAS"
    if "PAS" in n:
        return "PAS"
    if "CRAZ" in n:
        return "CRAZ"            
    return "OTHER"

payload["network"] = payload["name"].apply(infer_network)


payload["last_seen_utc"] = (
    payload["last_seen_utc"]
    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
)

response = supabase.table("purpleair_sensors_meta") \
    .upsert(payload.to_dict("records"), on_conflict="sensor_index") \
    .execute()

print("Supabase response:", response)
print(f"Attempted to upsert {len(payload)} sensors.")
