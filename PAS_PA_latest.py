import os
import requests
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

# Load your static sensor list
sensor_df = pd.read_csv("data/PAS_sensors.csv")
sensor_ids = sensor_df["sensor_index"].dropna().astype(int).tolist()
sensor_id_str = ",".join(map(str, sensor_ids))

# Build API call
url = "https://api.purpleair.com/v1/sensors"
headers = {"X-API-Key": os.getenv("PURPLEAIR_API_KEY")}
params = {
    "fields": "sensor_index,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b",
    "show_only": sensor_id_str
}

response = requests.get(url, headers=headers, params=params)
data = response.json()
fields = data["fields"]
rows = data["data"]
df_live = pd.DataFrame(rows, columns=fields)

# Merge with static sensor metadata
df = pd.merge(sensor_df, df_live, on="sensor_index", how="inner")

# Filter out sensors older than 3 hours
now = datetime.now(timezone.utc)
df["last_seen"] = pd.to_datetime(df["last_seen"], unit="s", utc=True)
df = df[df["last_seen"] >= (now - timedelta(hours=3))]

# Robust PM2.5 calculation (R logic ported)
def get_best_pm(a, b, avg):
    if pd.isna(a) and not pd.isna(b) and b <= 2000:
        return b
    if pd.isna(b) and not pd.isna(a) and a <= 2000:
        return a
    if a > 2000 and b <= 2000:
        return b
    if b > 2000 and a <= 2000:
        return a
    if not pd.isna(a) and not pd.isna(b):
        diff = abs(a - b)
        if diff > 50 and diff <= 500:
            return max(a, b)
        elif diff > 500:
            return None
        elif diff <= 50 and not pd.isna(avg) and avg >= 0:
            return avg
    return avg


df["pm_raw"] = df.apply(lambda x: get_best_pm(x["pm2.5_atm_a"], x["pm2.5_atm_b"], x["pm2.5_atm"]), axis=1)

# Apply RH correction
def correct_pm25(pm, rh):
    if pd.isna(pm): return None
    if pd.isna(rh): rh = 50
    if rh < 30:
        return pm / (1 + 0.24 / (100 / 30 - 1))
    elif rh > 70:
        return pm / (1 + 0.24 / (100 / 70 - 1))
    else:
        return pm / (1 + 0.24 / (100 / rh - 1))



# Color assignment
def get_color(pm, name):
    if "PAS" not in str(name):
        return "#808080"  # gray for non-PAS sensors
    if pd.isna(pm): return "#808080"
    if pm > 100: return "#640100"
    elif pm > 90: return "#9a0100"
    elif pm > 80: return "#cc0001"
    elif pm > 70: return "#fe0002"
    elif pm > 60: return "#fd6866"
    elif pm > 50: return "#ff9835"
    elif pm > 40: return "#ffcb00"
    elif pm > 30: return "#fffe03"
    elif pm > 20: return "#016797"
    elif pm > 10: return "#0099cb"
    else: return "#01cbff"


df["color"] = df.apply(lambda x: get_color(x["pm_corr"], x["name"]), axis=1)

# Clean result
result = df[[
    "sensor_index", "name", "latitude", "longitude",
    "pm_raw", "humidity", "pm_corr", "color", "last_seen"
]]

# Save as JSON for Leaflet or web app
result["last_seen"] = result["last_seen"].dt.strftime('%Y-%m-%d %H:%M:%S')
result.to_json("data/PAS_PM25_map.json", orient="records", indent=2)
