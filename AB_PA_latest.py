import os
import requests
import pandas as pd
import json
import pytz
from datetime import datetime, timezone, timedelta
import sys

# [Keep all your existing functions: get_best_pm, correct_pm25, get_color]

def get_alberta_sensors(api_key):
    """Get all sensors in Alberta using bounding box"""
    params = {
        "fields": "sensor_index,name,latitude,longitude",
        "nwlng": -120.0,
        "nwlat": 60.0,  
        "selng": -110.0,
        "selat": 49.0,
        "max_age": 86400
    }
    
    headers = {"X-API-Key": api_key}
    url = "https://api.purpleair.com/v1/sensors"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data or not data["data"]:
            print("No sensors found in Alberta bounding box")
            return []
            
        sensors = []
        for row in data["data"]:
            sensors.append({
                "sensor_index": row[0],
                "name": row[1],
                "latitude": row[2],
                "longitude": row[3]
            })
        
        print(f"Found {len(sensors)} sensors in Alberta")
        return sensors
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Alberta sensors: {e}")
        return []

def fetch_current_data(sensor_ids, api_key):
    """Fetch current sensor data"""
    headers = {"X-API-Key": api_key}
    sensor_id_str = ",".join(map(str, sensor_ids))
    
    params = {
        "fields": "sensor_index,last_seen,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b",
        "show_only": sensor_id_str
    }
    
    response = requests.get("https://api.purpleair.com/v1/sensors", headers=headers, params=params)
    data = response.json()
    fields = data["fields"]
    rows = data["data"]
    
    return pd.DataFrame(rows, columns=fields)

def process_data(df, sensor_metadata):
    """Process the sensor data"""
    df_processed = pd.merge(sensor_metadata, df, on="sensor_index", how="inner")
    
    # Filter out old sensors
    now = datetime.now(timezone.utc)
    df_processed["last_seen"] = pd.to_datetime(df_processed["last_seen"], unit="s", utc=True)
    df_processed = df_processed[df_processed["last_seen"] >= (now - timedelta(hours=3))]
    
    # Calculate PM values
    df_processed["pm_raw"] = df_processed.apply(
        lambda x: get_best_pm(x["pm2.5_atm_a"], x["pm2.5_atm_b"], x["pm2.5_atm"]), axis=1
    )
    df_processed["pm_corr"] = df_processed.apply(
        lambda x: correct_pm25(x["pm_raw"], x["humidity"]), axis=1
    )
    df_processed["color"] = df_processed.apply(
        lambda x: get_color(x["pm_corr"], x["name"]), axis=1
    )
    
    return df_processed

def main():
    api_key = os.getenv("PURPLEAIR_API_KEY")
    if not api_key:
        print("Error: PURPLEAIR_API_KEY environment variable not set")
        sys.exit(1)
    
    # Get all sensors in Alberta
    sensors = get_alberta_sensors(api_key)
    if not sensors:
        print("No sensors found. Exiting.")
        return
    
    sensor_df = pd.DataFrame(sensors)
    sensor_ids = sensor_df["sensor_index"].tolist()
    
    # Fetch and process current data
    df = fetch_current_data(sensor_ids, api_key)
    result = process_data(df, sensor_df)
    
    # Save current data
    ab_tz = pytz.timezone("America/Edmonton")
    result["last_seen"] = result["last_seen"].dt.tz_convert(ab_tz).dt.strftime('%Y-%m-%d %I:%M:%S %p')
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    result.to_json("data/AB_PM25_map.json", orient="records", indent=2)
    print(f"Data saved for {len(result)} sensors")

if __name__ == "__main__":
    main()
