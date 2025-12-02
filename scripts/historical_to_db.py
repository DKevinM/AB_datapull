#!/usr/bin/env python3
"""
Historical PurpleAir data collector
Pulls historical data and pushes directly to Supabase database
"""

import os
import requests
import pandas as pd
import sys
import argparse
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
import time

# Import your existing functions if they're in a module
# Or copy them here
def get_best_pm(a, b, avg):
    # [Copy your existing function]
    pass

def correct_pm25(pm, rh):
    # [Copy your existing function]
    pass

def push_to_supabase(records):
    """Push records to Supabase database"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            print("Missing Supabase credentials")
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        if records:
            # Insert in batches to avoid timeout
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                response = supabase.table("sensor_readings").insert(batch).execute()
                print(f"Pushed batch {i//batch_size + 1}: {len(batch)} records")
                time.sleep(1)  # Rate limiting
            
            print(f"✅ Total {len(records)} records pushed to Supabase")
            return True
            
    except Exception as e:
        print(f"❌ Error pushing to Supabase: {e}")
    
    return False

def fetch_sensor_historical_data(sensor_id, api_key, start_ts, end_ts):
    """Fetch historical data for a single sensor"""
    url = f"https://api.purpleair.com/v1/sensors/{sensor_id}/history"
    
    params = {
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "average": 60,  # 1 hour averages
        "fields": "time_stamp,humidity,pm2.5_atm,pm2.5_atm_a,pm2.5_atm_b"
    }
    
    headers = {"X-API-Key": api_key}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        records = []
        if "data" in data:
            for row in data["data"]:
                # Process each reading
                pm_raw = get_best_pm(row[3], row[4], row[2])  # Adjust indices based on API response
                pm_corr = correct_pm25(pm_raw, row[1]) if pm_raw else None
                
                record = {
                    "sensor_index": sensor_id,
                    "pm_raw": pm_raw,
                    "pm_corrected": pm_corr,
                    "humidity": row[1],
                    "recorded_at": datetime.fromtimestamp(row[0], tz=timezone.utc).isoformat()
                    # Add name, latitude, longitude from your sensor CSV if needed
                }
                records.append(record)
        
        return records
        
    except Exception as e:
        print(f"Error fetching sensor {sensor_id}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Collect historical PurpleAir data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    api_key = os.getenv("PURPLEAIR_API_KEY")
    if not api_key:
        print("Error: PURPLEAIR_API_KEY not set")
        sys.exit(1)
    
    # Load your sensor list
    try:
        sensor_df = pd.read_csv("data/AB_PA_sensors.csv")
        sensor_ids = sensor_df["sensor_index"].dropna().astype(int).tolist()
        print(f"Loaded {len(sensor_ids)} sensors")
    except FileNotFoundError:
        print("Error: data/AB_PA_sensors.csv not found")
        sys.exit(1)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    
    all_records = []
    
    # Fetch data for each sensor
    for sensor_id in sensor_ids:
        print(f"Fetching historical data for sensor {sensor_id}...")
        records = fetch_sensor_historical_data(sensor_id, api_key, start_ts, end_ts)
        all_records.extend(records)
        print(f"  Got {len(records)} records")
        time.sleep(2)  # Rate limiting between sensors
    
    # Push to Supabase
    if all_records:
        print(f"\nTotal records collected: {len(all_records)}")
        push_to_supabase(all_records)
    else:
        print("No records collected")

if __name__ == "__main__":
    main()
