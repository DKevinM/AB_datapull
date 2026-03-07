#!/usr/bin/env python3
# scripts/build_eaqhi.py

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

LAST6H_CSV = DATA_DIR / "last6h.csv"
PURPLE_JSON = DATA_DIR / "AB_PM25_map.json"
OUTPUT_JSON = DATA_DIR / "eAQHI_map.json"

# settings
PURPLE_RADIUS_KM = 20.0
MIN_PURPLE_SENSORS = 1
MAX_PURPLE_SENSORS = 4


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def compute_aqhi(o3_ppb, no2_ppb, pm25_ugm3):
    """
    AQHI formula using 3-hour averages.
    Returns integer 1-10, capped at 10.
    """
    val = (1000.0 / 10.4) * (
        math.exp(0.000537 * float(o3_ppb))
        + math.exp(0.000871 * float(no2_ppb))
        + math.exp(0.000487 * float(pm25_ugm3))
        - 3.0
    )
    aqhi = int(round(val))
    aqhi = max(1, min(aqhi, 10))
    return aqhi


def normalize_station_columns(df):
    rename_map = {}

    for col in df.columns:
        c = col.strip().lower()

        if c in ["stationname", "station_name", "station"]:
            rename_map[col] = "StationName"
        elif c in ["parametername", "parameter_name", "parameter"]:
            rename_map[col] = "ParameterName"
        elif c in ["readingdate", "datetime", "date", "timestamp"]:
            rename_map[col] = "ReadingDate"
        elif c in ["value", "val"]:
            rename_map[col] = "Value"
        elif c in ["latitude", "lat"]:
            rename_map[col] = "Latitude"
        elif c in ["longitude", "lon", "lng"]:
            rename_map[col] = "Longitude"
        elif c in ["units", "unit"]:
            rename_map[col] = "Units"
        elif c in ["shortform", "short_form", "short"]:
            rename_map[col] = "Shortform"

    df = df.rename(columns=rename_map)
    return df


def normalize_purple_records(records):
    rows = []

    for rec in records:
        lat = rec.get("lat", rec.get("Latitude", rec.get("latitude")))
        lon = rec.get("lon", rec.get("Longitude", rec.get("longitude")))
        pm = rec.get("pm_corr", rec.get("PM2.5", rec.get("pm25")))
        sensor_index = rec.get("sensor_index", rec.get("sensor"))
        name = rec.get("name", rec.get("label", "PurpleAir"))

        try:
            lat = float(lat)
            lon = float(lon)
            pm = float(pm)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(lat) or not np.isfinite(lon) or not np.isfinite(pm):
            continue

        rows.append({
            "sensor_index": sensor_index,
            "name": name,
            "lat": lat,
            "lon": lon,
            "pm_corr": pm
        })

    return pd.DataFrame(rows)


def read_purpleair_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        records = obj["data"]
    else:
        records = []

    return normalize_purple_records(records)


def standardize_parameter_name(x):
    s = str(x).strip().lower()

    mapping = {
        "nitrogen dioxide": "NO2",
        "no2": "NO2",
        "ozone": "O3",
        "o3": "O3",
        "fine particulate matter": "PM25",
        "pm2.5": "PM25",
        "pm25": "PM25",
        "aqhi": "AQHI",
    }

    return mapping.get(s, str(x).strip())


def build_station_wide_table(last6h):
    df = last6h.copy()
    df["ParameterName"] = df["ParameterName"].apply(standardize_parameter_name)
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce", utc=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    df = df.dropna(subset=["StationName", "ParameterName", "ReadingDate", "Value"])
    df = df.sort_values(["StationName", "ReadingDate"])

    # keep only parameters we care about
    df = df[df["ParameterName"].isin(["NO2", "O3", "PM25", "AQHI"])].copy()

    # wide table by timestamp
    wide = (
        df.pivot_table(
            index=["StationName", "ReadingDate", "Latitude", "Longitude"],
            columns="ParameterName",
            values="Value",
            aggfunc="mean"
        )
        .reset_index()
    )

    return wide


def get_station_candidates(wide):
    """
    Keep stations that have NO2 and O3 but do not have PM25 available.
    """
    out = []

    for station, g in wide.groupby("StationName"):
        g = g.sort_values("ReadingDate").copy()

        has_no2 = g["NO2"].notna().any() if "NO2" in g.columns else False
        has_o3 = g["O3"].notna().any() if "O3" in g.columns else False
        has_pm25 = g["PM25"].notna().any() if "PM25" in g.columns else False

        if has_no2 and has_o3 and not has_pm25:
            out.append((station, g))

    return out


def nearest_purpleair_subset(st_lat, st_lon, purple_df):
    if purple_df.empty:
        return pd.DataFrame()

    tmp = purple_df.copy()
    tmp["distance_km"] = tmp.apply(
        lambda r: haversine_km(st_lat, st_lon, r["lat"], r["lon"]),
        axis=1
    )

    tmp = tmp[tmp["distance_km"] <= PURPLE_RADIUS_KM].sort_values("distance_km")
    return tmp.head(MAX_PURPLE_SENSORS)


def build_station_result(station_name, station_df, purple_df):
    station_df = station_df.sort_values("ReadingDate").copy()

    lat = station_df["Latitude"].dropna().iloc[0] if station_df["Latitude"].notna().any() else np.nan
    lon = station_df["Longitude"].dropna().iloc[0] if station_df["Longitude"].notna().any() else np.nan

    if not np.isfinite(lat) or not np.isfinite(lon):
        return None

    nearby = nearest_purpleair_subset(lat, lon, purple_df)
    if len(nearby) < MIN_PURPLE_SENSORS:
        return None

    # single current PurpleAir average used as PM2.5 estimate
    pm25_est = nearby["pm_corr"].mean()

    # 3-hour rolling means from station gas data
    station_df["NO2_3h"] = station_df["NO2"].rolling(3, min_periods=3).mean()
    station_df["O3_3h"] = station_df["O3"].rolling(3, min_periods=3).mean()

    latest = station_df.dropna(subset=["NO2_3h", "O3_3h"]).sort_values("ReadingDate")
    if latest.empty:
        return None

    latest_row = latest.iloc[-1]

    # PM2.5 is assumed current-hour estimate repeated over 3 hours
    # first-pass operational version
    pm25_3h = float(pm25_est)

    aqhi_est = compute_aqhi(
        o3_ppb=float(latest_row["O3_3h"]),
        no2_ppb=float(latest_row["NO2_3h"]),
        pm25_ugm3=pm25_3h
    )

    sensors_used = nearby[["sensor_index", "name", "distance_km", "pm_corr"]].to_dict(orient="records")

    return {
        "station": station_name,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "AQHI": aqhi_est,
        "AQHI_type": "estimated",
        "pm25_source": "PurpleAir",
        "pm25_est": round(pm25_3h, 2),
        "o3_3h": round(float(latest_row["O3_3h"]), 2),
        "no2_3h": round(float(latest_row["NO2_3h"]), 2),
        "timestamp_utc": pd.Timestamp(latest_row["ReadingDate"]).isoformat(),
        "purpleair_sensor_count": int(len(nearby)),
        "purpleair_sensors": sensors_used
    }


def main():
    if not LAST6H_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {LAST6H_CSV}")

    if not PURPLE_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {PURPLE_JSON}")

    last6h = pd.read_csv(LAST6H_CSV)
    last6h = normalize_station_columns(last6h)

    required = {"StationName", "ParameterName", "ReadingDate", "Value", "Latitude", "Longitude"}
    missing = required - set(last6h.columns)
    if missing:
        raise ValueError(f"last6h.csv is missing required columns: {sorted(missing)}")

    purple_df = read_purpleair_json(PURPLE_JSON)
    if purple_df.empty:
        print("No usable PurpleAir records found.")
        OUTPUT_JSON.write_text("[]", encoding="utf-8")
        return

    wide = build_station_wide_table(last6h)
    candidates = get_station_candidates(wide)

    results = []
    for station_name, station_df in candidates:
        result = build_station_result(station_name, station_df, purple_df)
        if result is not None:
            results.append(result)

    results = sorted(results, key=lambda x: x["station"])

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} estimated AQHI stations to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
