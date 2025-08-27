import os
import requests
import pandas as pd
import json
import pytz
from datetime import datetime, timezone, timedelta
from math import ceil
from pathlib import Path
# before writing
Path("data").mkdir(parents=True, exist_ok=True)

API_URL = "https://api.purpleair.com/v1/sensors"

# ---------- Helpers ----------
def get_best_pm(a, b, avg):
    if pd.isna(a) and not pd.isna(b) and b <= 2000: return b
    if pd.isna(b) and not pd.isna(a) and a <= 2000: return a
    if a is not None and b is not None:
        if a > 2000 and b <= 2000: return b
        if b > 2000 and a <= 2000: return a
        diff = abs(a - b)
        if diff > 500: return None
        if diff > 50:  return max(a, b)
        if diff <= 50 and not pd.isna(avg) and avg >= 0: return avg
    return avg

def correct_pm25(pm, rh):
    if pd.isna(pm): return None
    if pd.isna(rh): rh = 50
    return pm / (1 + 0.24 / (100/max(min(rh,99.9),0.1) - 1))

def get_color(pm, name):
    if pd.isna(pm): return "#808080"
    v = float(pm)
    if   v > 100: return "#640100"
    elif v > 90:  return "#9a0100"
    elif v > 80:  return "#cc0001"
    elif v > 70:  return "#fe0002"
    elif v > 60:  return "#fd6866"
    elif v > 50:  return "#ff9835"
    elif v > 40:  return "#ffcb00"
    elif v > 30:  return "#fffe03"
    elif v > 20:  return "#016797"
    elif v > 10:  return "#0099cb"
    else:         return "#01cbff"

def fetch_chunk(sensor_ids, api_key):
    """Fetch a chunk of sensor indices. Returns DataFrame or empty DF."""
    headers = {"X-API-Key": api_key}
    params = {
        # comma-separated field names per PurpleAir v1 docs
        "fields": "sensor_index,name,latitude,longitude,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b",
        "show_only": ",".join(map(str, sensor_ids))
    }
    r = requests.get(API_URL, headers=headers, params=params, timeout=30)
    # 1) HTTP error?
    if r.status_code != 200:
        print(f"[ERROR] HTTP {r.status_code}: {r.text[:300]}")
        return pd.DataFrame()
    # 2) Try JSON parse
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("[ERROR] Non-JSON response:", r.text[:300])
        return pd.DataFrame()
    # 3) PurpleAir error payload?
    if isinstance(data, dict) and "error" in data:
        print("[ERROR] API error:", data.get("error"))
        return pd.DataFrame()
    # 4) Expected keys present?
    if not isinstance(data, dict) or "fields" not in data or "data" not in data:
        print("[ERROR] Unexpected payload keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        print("Payload preview:", str(data)[:300])
        return pd.DataFrame()
    fields = data["fields"]
    rows = data["data"]
    try:
        df_live = pd.DataFrame(rows, columns=fields)
    except Exception as e:
        print("[ERROR] Building DataFrame:", e)
        print("Fields:", fields)
        print("First row preview:", rows[0] if rows else None)
        return pd.DataFrame()
    return df_live

# ---------- Main ----------
def main():
    api_key = os.getenv("PURPLEAIR_API_KEY")
    if not api_key:
        raise RuntimeError("PURPLEAIR_API_KEY is not set in the environment.")

    # Load static list
    sensor_df = pd.read_csv("data/PAZA_sensors.csv")
    if "sensor_index" not in sensor_df.columns:
        raise RuntimeError("PAZA_sensors.csv is missing 'sensor_index' column.")

    sensor_ids = sensor_df["sensor_index"].dropna().astype(int).tolist()
    if not sensor_ids:
        print("[WARN] No sensor indices found in PAZA_sensors.csv")
        return

    # If you have many sensor IDs, some APIs prefer batching (e.g., <=100 per request)
    BATCH = 100
    frames = []
    for i in range(0, len(sensor_ids), BATCH):
        chunk = sensor_ids[i:i+BATCH]
        df_live = fetch_chunk(chunk, api_key)
        if not df_live.empty:
            frames.append(df_live)

    if not frames:
        print("[WARN] No live data returned.")
        return

    df_live_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["sensor_index"])

    # Merge with metadata
    df = pd.merge(sensor_df, df_live_all, on="sensor_index", how="inner")

    # Guard missing columns
    required = ["last_seen", "humidity", "pm2.5_atm", "pm2.5_atm_a", "pm2.5_atm_b", "name"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # Filter last_seen (within 3 hours)
    now = datetime.now(timezone.utc)
    df["last_seen"] = pd.to_datetime(df["last_seen"], unit="s", utc=True, errors="coerce")
    df = df[df["last_seen"] >= (now - timedelta(hours=3))]

    if df.empty:
        print("[WARN] After last_seen filter, no active sensors.")
        # Still write an empty JSON so downstream doesn’t crash
        pd.DataFrame(columns=[
            "sensor_index","name","latitude","longitude","humidity","pm_corr","color","last_seen"
        ]).to_json("data/PAZA_PM25_map.json", orient="records", indent=2)
        return

    # Compute PM and color
    df["pm_raw"] = df.apply(lambda x: get_best_pm(x.get("pm2.5_atm_a"), x.get("pm2.5_atm_b"), x.get("pm2.5_atm")), axis=1)
    df["pm_corr"] = df.apply(lambda x: correct_pm25(x["pm_raw"], x["humidity"]), axis=1)
    df["color"]   = df.apply(lambda x: get_color(x["pm_corr"], x["name"]), axis=1)

    # Prepare output
    out = df[["sensor_index","name","latitude","longitude","humidity","pm_corr","color","last_seen"]].copy()
    ab_tz = pytz.timezone("America/Edmonton")
    out["last_seen"] = out["last_seen"].dt.tz_convert(ab_tz).dt.strftime('%Y-%m-%d %I:%M:%S %p')

    out_path = Path("data/PAZA_PM25_map.json")
    if df.empty:
        out_path.write_text("[]", encoding="utf-8")
    else:
        out.to_json(out_path, orient="records", indent=2)

if __name__ == "__main__":
    main()
