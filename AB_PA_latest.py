import os
import requests
import pandas as pd
import json
import pytz
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
import sys

# Robust PM2.5 calculation (R logic ported) - ADD THESE FUNCTIONS
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
            return min(a, b)
        elif diff > 500:
            return None
        elif diff <= 50 and not pd.isna(avg) and avg >= 0:
            return avg
    return avg

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
    ## if "ACA" not in str(name):
    ##     return "#808080"  # gray for non-ACA sensors
    if pd.isna(pm): return "#808080"
    if pm > 100: return "#640100"  #AQHI 10+
    elif pm > 90: return "#9a0100" #AQHI 10
    elif pm > 80: return "#cc0001" #AQHI 9
    elif pm > 70: return "#fe0002" #AQHI 8
    elif pm > 60: return "#fd6866" #AQHI 7
    elif pm > 50: return "#ff9835" #AQHI 6
    elif pm > 40: return "#ffcb00" #AQHI 5
    elif pm > 30: return "#fffe03" #AQHI 4
    elif pm > 20: return "#016797" #AQHI 3
    elif pm > 10: return "#0099cb" #AQHI 2
    elif pm > 0: return "#01cbff"  #AQHI 1
    else: return "#D3D3D3"



def push_to_supabase(df_result):
    """Push sensor data to Supabase database"""
    try:
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_DB_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            print("Missing Supabase credentials. Skipping database upload.")
            return False
        
        # Create client
        supabase: Client = create_client(supabase_url, supabase_key)

        
        # ADD THIS: Create hourly timestamp
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hourly_timestamp = now.replace(minute=0, second=0, microsecond=0)
        
        
        # Prepare records for database
        records = []
        for _, row in df_result.iterrows():
            # Calculate pm_raw if not present
            if 'pm_raw' not in row or pd.isna(row['pm_raw']):
                pm_raw = get_best_pm(row["pm2.5_atm_a"], row["pm2.5_atm_b"], row["pm2.5_atm"])
            else:
                pm_raw = row['pm_raw']

            pm_corr = correct_pm25(pm_raw, row["humidity"]) if pd.notna(pm_raw) else None

            
            method = "avg"
            if pd.isna(row["pm2.5_atm_a"]) and not pd.isna(row["pm2.5_atm_b"]):
                method = "b_only"
            elif pd.isna(row["pm2.5_atm_b"]) and not pd.isna(row["pm2.5_atm_a"]):
                method = "a_only"
            elif abs(row["pm2.5_atm_a"] - row["pm2.5_atm_b"]) > 50:
                method = "large_diff"

            
            record = {
                "sensor_index": int(row["sensor_index"]),
                "recorded_at": hourly_timestamp.isoformat(),
            
                # Raw channels
                "pm25_atm_a": float(row["pm2.5_atm_a"]) if pd.notna(row["pm2.5_atm_a"]) else None,
                "pm25_atm_b": float(row["pm2.5_atm_b"]) if pd.notna(row["pm2.5_atm_b"]) else None,
                "humidity": float(row["humidity"]) if pd.notna(row["humidity"]) else None,
            
                # Derived values (unchanged logic)
                "pm_raw": float(pm_raw) if pd.notna(pm_raw) else None,
                "pm_corrected": float(pm_corr) if pd.notna(pm_corr) else None,
                "pm_method": method,
            
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            records.append(record)
        
        # Insert into database
        if records:
            response = supabase.table("sensor_readings").upsert(records).execute()
            print(f"Successfully pushed {len(records)} records to Supabase")

            print(f"Supabase response: {response}")
            
            return True

    
    except Exception as e:
        print(f"Error pushing to Supabase: {e}")
        print(f"Error type: {type(e).__name__}")
    
    return False





def main():


    print("=== DEBUG: CHECKING ENVIRONMENT ===")
    print(f"1. SUPABASE_DB_URL: {'[SET]' if os.getenv('SUPABASE_DB_URL') else '[MISSING]'}")
    print(f"2. SUPABASE_SERVICE_KEY: {'[SET]' if os.getenv('SUPABASE_SERVICE_KEY') else '[MISSING]'}")
    print(f"3. PURPLEAIR_API_KEY: {'[SET]' if os.getenv('PURPLEAIR_API_KEY') else '[MISSING]'}")
    print("=== END DEBUG ===")
    

    
    api_key = os.getenv("PURPLEAIR_API_KEY")
    if not api_key:
        print("Error: PURPLEAIR_API_KEY environment variable not set")
        sys.exit(1)

    
    # Load your static sensor list from CSV
    try:
        sensor_df = pd.read_csv("data/AB_PA_sensors.csv")
        sensor_df["sensor_index"] = pd.to_numeric(sensor_df["sensor_index"], errors="coerce")
        sensor_df = sensor_df.dropna(subset=["sensor_index"])
        sensor_df["sensor_index"] = sensor_df["sensor_index"].astype("int64")
        print(f"Loaded {len(sensor_df)} sensors from CSV")
    except FileNotFoundError:
        print("Error: data/AB_PA_sensors.csv not found")
        sys.exit(1)

        
    # Load dead_list.csv and remove those sensors
    dead_sensor_ids = set()
    try:
        dead_df = pd.read_csv("data/dead_list.csv")
        dead_df["sensor_index"] = pd.to_numeric(dead_df["sensor_index"], errors="coerce")
        dead_df = dead_df.dropna(subset=["sensor_index"])
        dead_df["sensor_index"] = dead_df["sensor_index"].astype("int64")
        dead_sensor_ids = set(dead_df["sensor_index"].tolist())
        print(f"Loaded {len(dead_sensor_ids)} sensors from dead_list.csv: {dead_sensor_ids}")
    
        original_count = len(sensor_df)
        sensor_df = sensor_df[~sensor_df["sensor_index"].isin(dead_sensor_ids)]
        removed_count = original_count - len(sensor_df)
        print(f"Removed {removed_count} dead sensors. {len(sensor_df)} sensors remaining.")
    except FileNotFoundError:
        print("Warning: data/dead_list.csv not found, proceeding with all sensors")
    except Exception as e:
        print(f"Warning: Error reading dead_list.csv: {e}, proceeding with all sensors")


    channel_override = {}
    try:
        override_df = pd.read_csv("data/channel_override.csv")
        override_df["sensor_index"] = pd.to_numeric(override_df["sensor_index"], errors="coerce")
        override_df = override_df.dropna(subset=["sensor_index"])
        override_df["sensor_index"] = override_df["sensor_index"].astype("int64")
    
        channel_override = dict(zip(
            override_df["sensor_index"],
            override_df["force_channel"]
        ))
    
        print(f"Loaded {len(channel_override)} channel overrides")
    
    except FileNotFoundError:
        print("No channel_override.csv found")
    
    
   
    # Build sensor_ids **after** filtering
    sensor_ids = sensor_df["sensor_index"].astype("int64").tolist()
    print(f"Sensor IDs after dead-list filter: {sensor_ids}")
    
    if not sensor_ids:
        print("No sensors remaining after filtering. Exiting.")
        sys.exit(0)
    
    sensor_id_str = ",".join(map(str, sensor_ids))


    print(f"Original sensor count: {len(sensor_df)}")
    print(f"Dead sensor IDs: {dead_sensor_ids}")
    
    
    # Build API call for ONLY the sensors in your CSV
    url = "https://api.purpleair.com/v1/sensors"
    headers = {"X-API-Key": api_key}
    params = {
        "fields": "sensor_index,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b",
        "show_only": sensor_id_str
    }
    
    # Fetch data
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    
    fields = data["fields"]
    rows   = data["data"]
    df_live = pd.DataFrame(rows, columns=fields)
    
    # Make sure sensor_index is numeric and comparable
    df_live["sensor_index"] = pd.to_numeric(df_live["sensor_index"], errors="coerce")
    df_live = df_live.dropna(subset=["sensor_index"])
    df_live["sensor_index"] = df_live["sensor_index"].astype("int64")
    
    print(f"Live dataframe from PurpleAir has {len(df_live)} rows")
    
    # Inner-join with sensor_df so only whitelisted & non-dead sensors remain
    meta_cols = ["sensor_index", "name", "latitude", "longitude"]
    missing_meta = [c for c in meta_cols if c not in sensor_df.columns]
    if missing_meta:
        raise ValueError(f"Missing columns in sensor_df: {missing_meta}")
    
    df = df_live.merge(
        sensor_df[meta_cols],
        on="sensor_index",
        how="inner",
        validate="many_to_one"
    )

    # EXPLICITLY filter out any dead sensors that slipped through
    original_len = len(df)
    df = df[~df["sensor_index"].isin(dead_sensor_ids)]
    if len(df) < original_len:
        print(f"Explicitly removed {original_len - len(df)} dead sensors after merge")

    
    # After filtering
    print(f"Sensor IDs after filter: {sensor_ids}")
    print(f"Are dead sensors still in sensor_ids? {[id for id in sensor_ids if id in dead_sensor_ids]}")
    
    # After API call
    print(f"API returned {len(df_live)} sensors")
    print(f"API sensor IDs: {sorted(df_live['sensor_index'].unique())}")
    print(f"Dead sensors from API: {[id for id in df_live['sensor_index'].unique() if id in dead_sensor_ids]}")
    
    # After merge
    print(f"After merge: {len(df)} sensors")
    print(f"Unique sensors after merge: {sorted(df['sensor_index'].unique())}")

    
    if df.empty:
        print("No sensors with valid metadata found after merge. Exiting.")
        return


    
    # Filter out sensors older than 3 hours
    now = datetime.now(timezone.utc)
    hourly_timestamp = now.replace(minute=0, second=0, microsecond=0)
    df["last_seen"] = pd.to_datetime(df["last_seen"], unit="s", utc=True)
    df = df[df["last_seen"] >= (now - timedelta(hours=3))]
    print(f"After time filter: {len(df)} sensors")


    # -------- PM Selection With Override + Safer Diff Logic --------
    def select_pm(row):
        sid = row["sensor_index"]
    
        a = row["pm2.5_atm_a"]
        b = row["pm2.5_atm_b"]
        avg = row["pm2.5_atm"]
    
        # 1️⃣ Forced OFF
        if sid in channel_override and channel_override[sid] == "OFF":
            return None, "off"
    
        # 2️⃣ Forced channel
        if sid in channel_override:
            if channel_override[sid] == "A":
                return a, "forced_A"
            if channel_override[sid] == "B":
                return b, "forced_B"
    
        # 3️⃣ Automatic logic
        if pd.notna(a) and pd.notna(b):
            diff = abs(a - b)
    
            # Extreme divergence → reject
            if diff > 500:
                return None, "extreme_diff"
    
            # Moderate divergence → choose LOWER (safer than max)
            if diff > 50:
                return min(a, b), "min_ab"
    
            # Small diff → use average
            return avg, "avg"
    
        # 4️⃣ One channel missing
        if pd.isna(a) and pd.notna(b):
            return b, "b_only"
        if pd.isna(b) and pd.notna(a):
            return a, "a_only"
    
        return avg, "fallback"
    
    
    # Apply selection
    df[["pm_raw", "pm_method"]] = df.apply(
        lambda x: pd.Series(select_pm(x)),
        axis=1
    )
    
    # Apply RH correction
    df["pm_corr"] = df.apply(
        lambda x: correct_pm25(x["pm_raw"], x["humidity"]),
        axis=1
    )

    
    # Clean result
    result = df.copy()
    
    # Save as JSON for Leaflet or web app
    ab_tz = pytz.timezone("America/Edmonton")
    result.loc[:, "last_seen"] = result["last_seen"].dt.tz_convert(ab_tz).dt.strftime('%Y-%m-%d %I:%M:%S %p')
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    result.to_json("data/AB_PM25_map.json", orient="records", indent=2)
    print(f"Final data saved for {len(result)} sensors to data/AB_PM25_map.json")

    print("Pushing data to Supabase...")
    push_to_supabase(result)



if __name__ == "__main__":
    main()
