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


# Load channel override
def load_channel_override():
    try:
        df = pd.read_csv("data/channel_override.csv")
        df["sensor_index"] = df["sensor_index"].astype(int)
        return dict(zip(df["sensor_index"], df["force_channel"]))
    except:
        return {}



def get_best_pm(a, b, avg):
    # Hard invalids
    if pd.isna(a) and pd.notna(b) and b <= 2000:
        return b
    if pd.isna(b) and pd.notna(a) and a <= 2000:
        return a
    if pd.notna(a) and a > 2000 and pd.notna(b) and b <= 2000:
        return b
    if pd.notna(b) and b > 2000 and pd.notna(a) and a <= 2000:
        return a

    if pd.notna(a) and pd.notna(b):
        diff = abs(a - b)

        # Extreme divergence → reject
        if diff > 500:
            return None

        # Moderate divergence → choose LOWER (safer)
        if diff > 50:
            return min(a, b)

        # Small difference → use average
        if pd.notna(avg):
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


def fetch_sensor_historical_data(
    sensor_id,
    api_key,
    start_ts,
    end_ts,
    sensor_metadata,
    channel_override
):


def push_to_supabase(records):
    """Push records to Supabase database"""
    try:
        supabase_url = os.getenv("SUPABASE_DB_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            print("Missing Supabase credentials")
            return False

        supabase: Client = create_client(supabase_url, supabase_key)

        if not records:
            print("No records to push.")
            return False

        batch_size = 100

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            response = (
                supabase
                .table("sensor_readings")
                .upsert(batch)
                .execute()
            )

            if hasattr(response, "error") and response.error:
                print("Supabase error:", response.error)
                return False

            print(f"Pushed batch {i//batch_size + 1}: {len(batch)} records")
            time.sleep(0.5)  # lighter throttle

        print(f"Total {len(records)} records pushed to Supabase")
        return True

    except Exception as e:
        print(f"Error pushing to Supabase: {e}")
        return False



def main():
    parser = argparse.ArgumentParser(description='Collect historical PurpleAir data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    channel_override = load_channel_override()

    api_key = os.getenv("PURPLEAIR_API_KEY")
    if not api_key:
        print("Error: PURPLEAIR_API_KEY not set")
        sys.exit(1)

    # Load sensor list
    try:
        sensor_df = pd.read_csv("data/AB_PA_sensors.csv")

        # Remove dead sensors before building sensor_ids
        try:
            dead_df = pd.read_csv("data/dead_list.csv")
            dead_df["sensor_index"] = dead_df["sensor_index"].astype(int)
            dead_ids = set(dead_df["sensor_index"].tolist())
            print(f"Loaded {len(dead_ids)} dead sensors")

            sensor_df = sensor_df[~sensor_df["sensor_index"].isin(dead_ids)]
        except FileNotFoundError:
            print("No dead_list.csv found")

        sensor_ids = sensor_df["sensor_index"].dropna().astype(int).tolist()
        print(f"{len(sensor_ids)} sensors after dead filter")

        sensor_metadata = sensor_df.set_index("sensor_index")[["name", "latitude", "longitude"]].to_dict('index')

    except FileNotFoundError:
        print("Error: data/AB_PA_sensors.csv not found")
        sys.exit(1)

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    all_records = []

    for sensor_id in sensor_ids:
        print(f"Fetching historical data for sensor {sensor_id}...")

        records = fetch_sensor_historical_data(
            sensor_id,
            api_key,
            start_ts,
            end_ts,
            sensor_metadata,
            channel_override
        )

        all_records.extend(records)
        print(f"  Got {len(records)} records")
        time.sleep(1)

    if all_records:
        print(f"\nTotal records collected: {len(all_records)}")
        push_to_supabase(all_records)
    else:
        print("No records collected")


if __name__ == "__main__":
    main()
