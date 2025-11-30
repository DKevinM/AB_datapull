#!/usr/bin/env python3
import os
import requests
import pandas as pd
import pytz
from datetime import datetime, timezone, timedelta
import sys
import time

# =========================================================
# CONFIG
# =========================================================

PURPLEAIR_API_KEY = os.getenv("PURPLEAIR_API_KEY")
if not PURPLEAIR_API_KEY:
    raise RuntimeError("Missing PURPLEAIR_API_KEY")

SENSOR_LIST_PATH = "data/AB_PA_sensors.csv"
HISTORY_PATH = "data/AB_PA_history_full.csv"
LIVE_JSON_PATH = "data/AB_PM25_map.json"

EDM_TZ = pytz.timezone("America/Edmonton")

# =========================================================
# PM LOGIC
# =========================================================

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


def correct_pm25(pm, rh):
    if pd.isna(pm):
        return None
    if pd.isna(rh):
        rh = 50
    if rh < 30:
        return pm / (1 + 0.24 / (100/30 - 1))
    elif rh > 70:
        return pm / (1 + 0.24 / (100/70 - 1))
    else:
        return pm / (1 + 0.24 / (100/rh - 1))


def get_color(pm, name):
    # if "ACA" not in str(name):
    #    return "#808080"
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

# =========================================================
# API: FETCH HISTORY FOR A SENSOR
# =========================================================

def fetch_history(sensor_id, start_ts, end_ts):
    url = f"https://api.purpleair.com/v1/sensors/{sensor_id}/history"
    params = {
        "average": 0,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "fields": "time_stamp,humidity,pm2.5_atm"
    }
    headers = {"X-API-Key": PURPLEAIR_API_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    fields = data.get("fields", [])
    rows = data.get("data", [])
    df = pd.DataFrame(rows, columns=fields)
    df["sensor_index"] = sensor_id
    return df


# =========================================================
# MAIN FULL HISTORY MODE
# =========================================================

def run_full_history():
    print("🔵 Running FULL history pull (Jan 1, 2025 → now)...")

    start_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.now(timezone.utc).timestamp())

    sensors = pd.read_csv(SENSOR_LIST_PATH)
    ids = sensors["sensor_index"].dropna().astype(int).tolist()

    all_frames = []

    for sid in ids:
        print(f"→ Fetching {sid} ...")
        try:
            df = fetch_history(sid, start_ts, end_ts)
            if df.empty:
                print(f"   No data.")
                continue
            df["pm_corr"] = df.apply(lambda r: correct_pm25(r["pm2.5_atm"], r["humidity"]), axis=1)
            all_frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"   ERROR: {e}")

    if not all_frames:
        print("No history returned.")
        return

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(HISTORY_PATH, index=False)

    print(f"✅ Saved full history → {HISTORY_PATH}")


# =========================================================
# MAIN LIVE UPDATE MODE (YOUR ORIGINAL LOGIC)
# =========================================================

def run_live_update():

    print("🟢 Running 30-minute live update...")

    sensor_df = pd.read_csv(SENSOR_LIST_PATH)
    sensor_ids = sensor_df["sensor_index"].dropna().astype(int).tolist()
    id_str = ",".join(map(str, sensor_ids))

    url = "https://api.purpleair.com/v1/sensors"
    headers = {"X-API-Key": PURPLEAIR_API_KEY}
    params = {
        "fields": "sensor_index,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b",
        "show_only": id_str
    }

    r = requests.get(url, headers=headers, params=params)
    data = r.json()

    df_live = pd.DataFrame(data["data"], columns=data["fields"])
    df = pd.merge(sensor_df, df_live, on="sensor_index", how="inner")

    now = datetime.now(timezone.utc)
    df["last_seen"] = pd.to_datetime(df["last_seen"], unit="s", utc=True)
    df = df[df["last_seen"] >= (now - timedelta(hours=3))]

    df["pm_raw"] = df.apply(lambda x: get_best_pm(x["pm2.5_atm_a"], x["pm2.5_atm_b"], x["pm2.5_atm"]), axis=1)
    df["pm_corr"] = df.apply(lambda x: correct_pm25(x["pm_raw"], x["humidity"]), axis=1)
    df["color"] = df.apply(lambda x: get_color(x["pm_corr"], x["name"]), axis=1)

    result = df[[
        "sensor_index", "name", "latitude", "longitude",
        "humidity", "pm_corr", "color", "last_seen"
    ]]

    result["last_seen"] = result["last_seen"].dt.tz_convert(EDM_TZ).dt.strftime('%Y-%m-%d %I:%M:%S %p')
    result.to_json(LIVE_JSON_PATH, orient="records", indent=2)

    print(f"✅ Saved live map → {LIVE_JSON_PATH}")

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    if "--full" in sys.argv:
        run_full_history()
    else:
        run_live_update()
