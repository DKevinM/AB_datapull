import os
from supabase import create_client, Client
import pandas as pd
from datetime import datetime

class SupabaseDB:
    def __init__(self):
        self.url = os.getenv("SUPABASE_DB_URL")
        self.key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for write access
        self.supabase: Client = create_client(self.url, self.key)
    
    def insert_sensor_readings(self, sensor_data):
        """Insert sensor readings into the database"""
        try:
            # Convert DataFrame to list of dictionaries
            if isinstance(sensor_data, pd.DataFrame):
                records = sensor_data.to_dict('records')
            else:
                records = sensor_data
            
            # Insert records
            response = self.supabase.table("sensor_readings").insert(records).execute()
            print(f"Inserted {len(records)} records")
            return response
        except Exception as e:
            print(f"Error inserting data: {e}")
            return None
    
    def get_latest_readings(self, limit=1000):
        """Get latest sensor readings"""
        try:
            response = self.supabase.table("sensor_readings")\
                .select("*")\
                .order("recorded_at", desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []
