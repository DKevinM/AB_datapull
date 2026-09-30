# PA_border_pull.py
# Pull outdoor PurpleAir sensors within BUFFER_KM of the Alberta border in a
# neighbouring jurisdiction and save as CSV (companion to PA_AB_pull.py).
#
#   python PA_border_pull.py BC   -> data/BC_PA_sensors.csv
#   python PA_border_pull.py NT   -> data/NT_PA_sensors.csv
#
# Added 2026-09-30 (BC first, NWT same day) so upwind smoke from the Elk
# Valley, Golden/Revelstoke, Valemount, the Peace and the Hay River / Fort
# Smith area shows up before it reaches AB. BC sizing at the time (outdoor
# sensors): 100 km = 34, 150 km = 85, 200 km = 178, 250 km = 262.
# SK is NOT pulled here: SK_datapull already collects all of SK (Supabase
# province='SK'); build_pa_regional_aqhi.py cuts the same band from its file.
# To widen a band, change BUFFER_KM only.

import os
import sys
import requests
import pandas as pd
import geopandas as gpd

from supabase import create_client

BUFFER_KM = 150

REGION = sys.argv[1].upper() if len(sys.argv) > 1 else "BC"
if REGION not in ("BC", "NT"):
    sys.exit(f"Unknown region {REGION} (use BC or NT)")

# 1) Alberta boundary, buffered in a metric CRS (Canada Atlas Lambert)
ab = gpd.read_file("data/Alberta.shp")
ab_m = ab.to_crs(epsg=3978).geometry.union_all()
band_m = ab_m.buffer(BUFFER_KM * 1000)
band_ll = gpd.GeoSeries([band_m], crs=3978).to_crs(epsg=4326)

# 2) bbox of the buffered area, clipped to the region's side of AB
minx, miny, maxx, maxy = band_ll.total_bounds
if REGION == "BC":
    maxx = -114.0
else:
    miny = 60.0

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

# 5) Keep the region only: inside the buffer and outside Alberta, then
#    BC = 49-60N west of 114W (east of AB is SK at 110W, south of 49N is the US)
#    NT = north of 60N (everything north of AB within 150 km is NWT)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
).to_crs(epsg=3978)

gdf["border_km"] = (gdf.geometry.distance(ab_m) / 1000).round(1)

in_band = gdf.geometry.within(band_m) & ~gdf.geometry.intersects(ab_m)
if REGION == "BC":
    in_region = gdf["latitude"].between(49, 60, inclusive="left") & (gdf["longitude"] < -114)
else:
    in_region = gdf["latitude"] >= 60
band = gdf[in_band & in_region].copy()
band["province"] = REGION

band_no_geom = pd.DataFrame(band.drop(columns="geometry"))

# Save only recently active sensors for downstream live PurpleAir pulls
band_live = band_no_geom[band_no_geom["active_30d"] == True].copy()
band_live.to_csv(f"data/{REGION}_PA_sensors.csv", index=False)

print(f"Total sensors from API: {len(gdf)}")
print(f"{REGION} sensors within {BUFFER_KM} km of AB: {len(band)} ({len(band_live)} active 30d)")

# 6) Push sensor metadata into Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

payload = band_no_geom[[
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

# No border networks yet; PA_AB_pull's name matching would mis-tag "Rogers Pass" as PAS
payload["network"] = "OTHER"

payload["last_seen_utc"] = payload["last_seen_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

response = supabase.table("purpleair_sensors_meta") \
    .upsert(payload.to_dict("records"), on_conflict="sensor_index") \
    .execute()

print(f"Attempted to upsert {len(payload)} sensors.")
