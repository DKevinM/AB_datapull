#!/usr/bin/env python3
# scripts/build_eaqhi.py

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import os


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

LAST6H_CSV = DATA_DIR / "last6h.csv"
PURPLE_HISTORY = DATA_DIR / "AB_PA_history.csv"
PURPLE_JSON = DATA_DIR / "AB_PM25_map.json"
OUTPUT_JSON = DATA_DIR / "eAQHI_map.json"

STATIONS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"

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


def read_purple_history(path):

    if not os.path.exists(path):
        print("No PurpleAir history file yet — creating empty dataset.")
        return pd.DataFrame(columns=[
            "sensor_index",
            "datetime",
            "pm_corr"
        ])

    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["pm_corr"] = pd.to_numeric(df["pm_corr"], errors="coerce")

    df = df.dropna(subset=["sensor_index", "datetime", "pm_corr"])

    return df
    

def compute_sensor_3h_means(history_df):
    now = history_df["datetime"].max()

    window_start = now - pd.Timedelta(hours=3)

    recent = history_df[history_df["datetime"] >= window_start]

    means = (
        recent.groupby("sensor_index")["pm_corr"]
        .mean()
        .reset_index()
        .rename(columns={"pm_corr": "pm25_3h"})
    )

    return means




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
    Permanent equipment-gap case (e.g. Woodcroft, Breton, Carrot Creek) —
    NOT the "station is fully offline" case, see get_offline_stations below.
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


def fetch_station_roster():
    """
    Full official station list (name + coordinates) from Alberta's Stations
    metadata endpoint — separate from the live measurements feed, so it's
    expected to stay available even during a measurements-side outage. Used
    only to detect which stations are fully offline (see get_offline_stations);
    never used as a source of readings.
    """
    try:
        r = requests.get(STATIONS_URL, params={"$format": "json"}, timeout=30)
        r.raise_for_status()
        rows = r.json().get("value", [])
    except Exception as ex:
        print(f"Could not fetch station roster ({type(ex).__name__}: {ex}); "
              f"skipping fully-offline-station detection this run.")
        return []

    roster = []
    for row in rows:
        try:
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        name = row.get("Name")
        if name:
            roster.append({"station": name, "lat": lat, "lon": lon})
    return roster


def get_offline_stations(wide, roster):
    """
    Roster stations with zero fresh readings of any tracked parameter —
    a station gone fully dark (e.g. a provincial data outage), as opposed
    to get_station_candidates' permanent PM2.5-sensor-gap case.
    """
    reporting = set(wide["StationName"].unique()) if not wide.empty else set()
    return [s for s in roster if s["station"] not in reporting]


def fetch_mds_wide_table(hours_back=3):
    """
    Second, independent official-grade data source: the airshed's own MDS
    telemetry (ACA_data_pipe / WCAS_data_pipe -> Supabase measurements),
    read back out through the api_measurements view. Same station/parameter
    identity as the government feed, but a completely separate pipe — so a
    station missing from last6h.csv (government outage, or this specific
    server down) may still be reporting here, and vice versa.

    Returns the same wide shape as build_station_wide_table so it can reuse
    the same downstream logic: StationName, ReadingDate, Latitude,
    Longitude, NO2, O3, PM25.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("SUPABASE_URL/SUPABASE_SERVICE_KEY not set; skipping MDS-direct source.")
        return pd.DataFrame(columns=["StationName", "ReadingDate", "Latitude", "Longitude", "NO2", "O3", "PM25"])

    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_back)).isoformat()
    endpoint = f"{url.rstrip('/')}/rest/v1/api_measurements"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    params = {
        "parameter_code": "in.(NO2,O3,PM25)",
        "reading_time": f"gte.{since}",
        "select": "StationName,Latitude,Longitude,parameter_code,value,reading_time",
        "limit": "10000",
    }
    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
    except Exception as ex:
        print(f"Could not fetch MDS-direct measurements ({type(ex).__name__}: {ex}); skipping this source.")
        return pd.DataFrame(columns=["StationName", "ReadingDate", "Latitude", "Longitude", "NO2", "O3", "PM25"])

    if not rows:
        return pd.DataFrame(columns=["StationName", "ReadingDate", "Latitude", "Longitude", "NO2", "O3", "PM25"])

    df = pd.DataFrame(rows).rename(columns={"reading_time": "ReadingDate", "parameter_code": "ParameterName", "value": "Value"})
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce", utc=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["StationName", "ParameterName", "ReadingDate", "Value"])

    wide = (
        df.pivot_table(
            index=["StationName", "ReadingDate", "Latitude", "Longitude"],
            columns="ParameterName",
            values="Value",
            aggfunc="mean"
        )
        .reset_index()
    )
    for c in ["NO2", "O3", "PM25"]:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide


def pm25_to_eaqhi(pm):
    """PM2.5-only estimated-AQHI proxy — same breakpoints used across the
    sit-rep repos' core/aqhi.py, for consistency. Used when a station is
    fully offline and there's no recent NO2/O3 to run the full formula."""
    if pm is None or not np.isfinite(pm):
        return None
    if pm <= 10: return 1
    if pm <= 20: return 2
    if pm <= 30: return 3
    if pm <= 40: return 4
    if pm <= 50: return 5
    if pm <= 60: return 6
    if pm <= 70: return 7
    if pm <= 80: return 8
    if pm <= 90: return 9
    if pm <= 100: return 10
    return 11


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
    
    nearby = nearby.copy()
    
    # -----------------------------------------
    # Step 1 — remove obvious sensor outliers
    # -----------------------------------------
    if len(nearby) >= 3:
        q1 = nearby["pm_corr"].quantile(0.25)
        q3 = nearby["pm_corr"].quantile(0.75)
        iqr = q3 - q1
    
        nearby = nearby[
            (nearby["pm_corr"] >= q1 - 1.5 * iqr) &
            (nearby["pm_corr"] <= q3 + 1.5 * iqr)
        ]
    
        if nearby.empty:
            return None
    
    
    # -----------------------------------------
    # Step 2 — distance weighting
    # closer sensors have larger influence
    # -----------------------------------------
    nearby["weight"] = 1 / nearby["distance_km"].clip(lower=0.5)
    
    pm25_weighted = np.average(
        nearby["pm_corr_3h"],
        weights=nearby["weight"]
    )
        
    
    # -----------------------------------------
    # Step 3 — approximate 3-hour PM2.5 average
    # assume PurpleAir value represents current hour
    # and smooth slightly to approximate short-term
    # variability
    # -----------------------------------------
    pm25_3h = float(pm25_weighted)

    
    # 3-hour rolling means from station gas data
    station_df["NO2_3h"] = station_df["NO2"].rolling(3, min_periods=3).mean()
    station_df["O3_3h"] = station_df["O3"].rolling(3, min_periods=3).mean()

    latest = station_df.dropna(subset=["NO2_3h", "O3_3h"]).sort_values("ReadingDate")
    if latest.empty:
        return None

    latest_row = latest.iloc[-1]

    # PM2.5 is assumed current-hour estimate repeated over 3 hours
    # first-pass operational version
    pm25_3h = float(pm25_weighted)

    o3_ppb = float(latest_row["O3_3h"]) * 1000
    no2_ppb = float(latest_row["NO2_3h"]) * 1000
    aqhi_est = compute_aqhi(
        o3_ppb=o3_ppb,
        no2_ppb=no2_ppb,
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
        "o3_3h": round(o3_ppb, 1),
        "no2_3h": round(no2_ppb, 1),
        "timestamp_utc": pd.Timestamp(latest_row["ReadingDate"]).isoformat(),
        "purpleair_sensor_count": int(len(nearby)),
        "purpleair_sensors": sensors_used
    }


def build_offline_station_result(station_name, lat, lon, purple_df):
    """PM2.5-only eAQHI estimate for a station with no fresh readings at
    all (so no NO2/O3 available for the full formula build_station_result
    uses) — same nearest-PurpleAir search and outlier/weighting logic,
    just converted via the PM2.5-only breakpoint table instead."""
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None

    nearby = nearest_purpleair_subset(lat, lon, purple_df)
    if len(nearby) < MIN_PURPLE_SENSORS:
        return None
    nearby = nearby.copy()

    if len(nearby) >= 3:
        q1 = nearby["pm_corr"].quantile(0.25)
        q3 = nearby["pm_corr"].quantile(0.75)
        iqr = q3 - q1
        nearby = nearby[
            (nearby["pm_corr"] >= q1 - 1.5 * iqr) &
            (nearby["pm_corr"] <= q3 + 1.5 * iqr)
        ]
        if nearby.empty:
            return None

    nearby["weight"] = 1 / nearby["distance_km"].clip(lower=0.5)
    pm25_3h = float(np.average(nearby["pm_corr_3h"], weights=nearby["weight"]))

    aqhi_est = pm25_to_eaqhi(pm25_3h)
    if aqhi_est is None:
        return None

    sensors_used = nearby[["sensor_index", "name", "distance_km", "pm_corr"]].to_dict(orient="records")

    return {
        "station": station_name,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "AQHI": aqhi_est,
        "AQHI_type": "estimated_pm25_only",
        "pm25_source": "PurpleAir",
        "pm25_est": round(pm25_3h, 2),
        "o3_3h": None,
        "no2_3h": None,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpleair_sensor_count": int(len(nearby)),
        "purpleair_sensors": sensors_used,
        "offline_reason": "no fresh reading of any parameter — station appears fully offline"
    }


def build_mds_direct_result(station_name, mds_station_df, purple_df):
    """
    Real reading from the airshed's own MDS telemetry, for a station that's
    gone dark in the government feed — higher confidence than
    build_offline_station_result's PM2.5-only estimate, since NO2/O3 (and
    often PM2.5 too) are genuine sensor values via a second independent
    pipe, not inferred from nearby PurpleAir. Only PM2.5 gets borrowed from
    PurpleAir, and only if this specific station's MDS feed has no PM2.5
    channel at all (e.g. Carrot Creek, Breton, Meadows, Wagner2).
    """
    mds_station_df = mds_station_df.sort_values("ReadingDate").copy()

    lat = mds_station_df["Latitude"].dropna().iloc[0] if mds_station_df["Latitude"].notna().any() else np.nan
    lon = mds_station_df["Longitude"].dropna().iloc[0] if mds_station_df["Longitude"].notna().any() else np.nan
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None

    mds_station_df["NO2_3h"] = mds_station_df["NO2"].rolling(3, min_periods=1).mean()
    mds_station_df["O3_3h"] = mds_station_df["O3"].rolling(3, min_periods=1).mean()

    latest = mds_station_df.dropna(subset=["NO2_3h", "O3_3h"]).sort_values("ReadingDate")
    if latest.empty:
        return None
    latest_row = latest.iloc[-1]
    # MDS reports NO2/O3 natively in ppb already (unlike the government
    # feed, which needs a ppm->ppb *1000 conversion in build_station_result)
    # — no conversion here.
    o3_ppb = float(latest_row["O3_3h"])
    no2_ppb = float(latest_row["NO2_3h"])

    pm25_source = None
    pm25_3h = None
    sensors_used = None

    pm25_latest = mds_station_df.dropna(subset=["PM25"]).sort_values("ReadingDate")
    if not pm25_latest.empty:
        pm25_3h = float(pm25_latest["PM25"].tail(3).mean())
        pm25_source = "MDS"
    else:
        nearby = nearest_purpleair_subset(lat, lon, purple_df)
        if len(nearby) < MIN_PURPLE_SENSORS:
            return None
        nearby = nearby.copy()
        if len(nearby) >= 3:
            q1 = nearby["pm_corr"].quantile(0.25)
            q3 = nearby["pm_corr"].quantile(0.75)
            iqr = q3 - q1
            nearby = nearby[(nearby["pm_corr"] >= q1 - 1.5 * iqr) & (nearby["pm_corr"] <= q3 + 1.5 * iqr)]
            if nearby.empty:
                return None
        nearby["weight"] = 1 / nearby["distance_km"].clip(lower=0.5)
        pm25_3h = float(np.average(nearby["pm_corr_3h"], weights=nearby["weight"]))
        pm25_source = "PurpleAir"
        sensors_used = nearby[["sensor_index", "name", "distance_km", "pm_corr"]].to_dict(orient="records")

    aqhi_est = compute_aqhi(o3_ppb=o3_ppb, no2_ppb=no2_ppb, pm25_ugm3=pm25_3h)

    result = {
        "station": station_name,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "AQHI": aqhi_est,
        "AQHI_type": "mds_direct",
        "pm25_source": pm25_source,
        "pm25_est": round(pm25_3h, 2),
        "o3_3h": round(o3_ppb, 1),
        "no2_3h": round(no2_ppb, 1),
        "timestamp_utc": pd.Timestamp(latest_row["ReadingDate"]).isoformat(),
        "offline_reason": "missing from government feed — using airshed's own MDS telemetry instead"
    }
    if sensors_used is not None:
        result["purpleair_sensor_count"] = len(sensors_used)
        result["purpleair_sensors"] = sensors_used
    return result


def push_estimates_to_supabase(results):
    """
    Feed estimated AQHI values into the same aqhi_data table the Cubist
    forecast model reads (run_cubist_forecast.py, SUPABASE_TABLE=aqhi_data)
    — so its lag features (AQHI_lag1..24) don't go NaN for a station just
    because the official feed has a gap, without changing that model's code
    at all: it already treats "AQHI" as "AQHI", estimated or not.

    Also writes a sentinel "AQHI_estimated"=1 row for the same
    (StationName, ReadingDate) so a training pipeline can exclude these —
    deliberately no schema change (direct DB access from this box is
    unreachable: Supabase's db.*.supabase.co host is IPv6-only and this
    box has no IPv6 route), just an additive row using the same long-format
    schema fetch_push.py already writes.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("SUPABASE_URL/SUPABASE_SERVICE_KEY not set; skipping Supabase push.")
        return

    records = []
    for r in results:
        if r.get("AQHI") is None:
            continue
        reading_date = pd.Timestamp(r["timestamp_utc"]).floor("h").isoformat().replace("+00:00", "Z")
        records.append({"StationName": r["station"], "ParameterName": "AQHI", "ReadingDate": reading_date, "Value": r["AQHI"]})
        records.append({"StationName": r["station"], "ParameterName": "AQHI_estimated", "ReadingDate": reading_date, "Value": 1})

    if not records:
        return

    endpoint = f"{url.rstrip('/')}/rest/v1/aqhi_data"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    params = {"on_conflict": "StationName,ParameterName,ReadingDate"}
    resp = requests.post(endpoint, headers=headers, params=params, data=json.dumps(records), timeout=60)
    if resp.status_code >= 400:
        print(f"Supabase push failed (status {resp.status_code}): {resp.text}")
    else:
        print(f"Pushed {len(records)} estimated AQHI/flag rows to Supabase for {len(results)} stations.")


def main():
    if not LAST6H_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {LAST6H_CSV}")

    if not PURPLE_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {PURPLE_JSON}")

    last6h = pd.read_csv(LAST6H_CSV)
    last6h = normalize_station_columns(last6h)

    purple_df = read_purpleair_json(PURPLE_JSON)

    purple_history = read_purple_history(PURPLE_HISTORY)
    sensor_means = compute_sensor_3h_means(purple_history)
    purple_df = purple_df.merge(sensor_means, on="sensor_index", how="left")
    # create a new column that prefers the 3-hour average
    purple_df["pm_corr_3h"] = purple_df["pm25_3h"].fillna(purple_df["pm_corr"])


    required = {"StationName", "ParameterName", "ReadingDate", "Value", "Latitude", "Longitude"}
    missing = required - set(last6h.columns)
    if missing:
        raise ValueError(f"last6h.csv is missing required columns: {sorted(missing)}")


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

    # Fully-offline stations (e.g. a provincial data outage) — separate from
    # the permanent PM2.5-sensor-gap case above, since there's no NO2/O3 to
    # run the full formula either. Try the airshed's own MDS telemetry first
    # (real sensor data via a second independent pipe) before falling back
    # to a PurpleAir-only estimate.
    roster = fetch_station_roster()
    if roster:
        offline = get_offline_stations(wide, roster)
        mds_wide = fetch_mds_wide_table()
        mds_by_station = {name: g for name, g in mds_wide.groupby("StationName")} if not mds_wide.empty else {}
        for s in offline:
            result = None
            mds_df = mds_by_station.get(s["station"])
            if mds_df is not None:
                result = build_mds_direct_result(s["station"], mds_df, purple_df)
            if result is None:
                result = build_offline_station_result(s["station"], s["lat"], s["lon"], purple_df)
            if result is not None:
                results.append(result)

    results = sorted(results, key=lambda x: x["station"])

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} estimated AQHI stations to {OUTPUT_JSON}")

    try:
        push_estimates_to_supabase(results)
    except Exception as ex:
        # Non-fatal: the map layer (OUTPUT_JSON) is already written above,
        # so a Supabase hiccup here shouldn't fail the whole cron run.
        print(f"Supabase push error ({type(ex).__name__}: {ex}); eAQHI_map.json was still written.")


if __name__ == "__main__":
    main()
