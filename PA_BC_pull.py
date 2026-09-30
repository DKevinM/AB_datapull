# PA_BC_pull.py
# Pull outdoor PurpleAir sensors in BC within BC_BUFFER_KM of the Alberta
# border and save as CSV (companion to PA_AB_pull.py).
#
# Added 2026-09-30: BC border band so upwind smoke from the Elk Valley,
# Golden/Revelstoke, Valemount and the Peace shows up before it reaches AB.
# Sizing at the time (outdoor sensors): 100 km = 34, 150 km = 85,
# 200 km = 178, 250 km = 262. To widen the band, change BC_BUFFER_KM only.

import os
import requests
import pandas as pd
import geopandas as gpd

from supabase import create_client

BC_BUFFER_KM = 150

# 1) Alberta boundary, buffered in a metric CRS (Canada Atlas Lambert)
ab = gpd.read_file("data/Alberta.shp")
ab_m = ab.to_crs(epsg=3978).geometry.union_all()
band_m = ab_m.buffer(BC_BUFFER_KM * 1000)
band_ll = gpd.GeoSeries([band_m], crs=3978).to_crs(epsg=4326)

# 2) bbox of the buffered area, clipped to the BC side (west of 114W)
minx, miny, maxx, maxy = band_ll.total_bounds
maxx = -114.0

# 3) Call PurpleAir /v1/sensors endpoint
url = "https://api.purpleair.com/v1/sensors"
headers = {"X-API-Key": os.getenv("PURPLEAIR_API_KEY")}

params = {
    "fields": "sensor_index,name,latitude,longitude,location_type,last_seen",
    "location_type": 0,
    "nwlng": minx,
    "nwlat": maxy,
    "selng": maxx,
    "selat": miny,
}

resp = requests.get(url, headers=headers, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

rows = data.get("data", [])
if not rows:
    raise RuntimeError("No sensors returned from PurpleAir – check bbox or API key.")

# 4) Convert to DataFrame
df = pd.DataFrame(rows, columns=data.get("fields", []))
for c in ["sensor_index", "latitude", "longitude", "last_seen"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["sensor_index", "latitude", "longitude"])
df["sensor_index"] = df["sensor_index"].astype("int64")

df["last_seen_utc"] = pd.to_datetime(df["last_seen"], unit="s", utc=True)
now_utc = pd.Timestamp.now(tz="UTC")
df["age_days"] = (now_utc - df["last_seen_utc"]).dt.total_seconds() / 86400
df["active_30d"] = df["age_days"] <= 30
df["active_7d"] = df["age_days"] <= 7

# 5) Keep BC only: inside the buffer, outside Alberta, 49-60N, west of 114W
#    (east of AB is SK at 110W, south of 49N is the US, north of 60N is NWT/YT)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
).to_crs(epsg=3978)

gdf["border_km"] = (gdf.geometry.distance(ab_m) / 1000).round(1)

bc = gdf[
    gdf.geometry.within(band_m)
    & ~gdf.geometry.intersects(ab_m)
    & gdf["latitude"].between(49, 60, inclusive="left")
    & (gdf["longitude"] < -114)
].copy()
bc["province"] = "BC"

bc_no_geom = pd.DataFrame(bc.drop(columns="geometry"))

# Save only recently active sensors for downstream live PurpleAir pulls
bc_live = bc_no_geom[bc_no_geom["active_30d"] == True].copy()
bc_live.to_csv("data/BC_PA_sensors.csv", index=False)

print(f"Total sensors from API: {len(gdf)}")
print(f"BC sensors within {BC_BUFFER_KM} km of AB: {len(bc)} ({len(bc_live)} active 30d)")

# 6) Push sensor metadata into Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

payload = bc_no_geom[[
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

# No BC networks yet; PA_AB_pull's name matching would mis-tag "Rogers Pass" as PAS
payload["network"] = "OTHER"

payload["last_seen_utc"] = payload["last_seen_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

response = supabase.table("purpleair_sensors_meta") \
    .upsert(payload.to_dict("records"), on_conflict="sensor_index") \
    .execute()

print(f"Attempted to upsert {len(payload)} sensors.")
