#!/usr/bin/env python3
# scripts/build_pa_regional_aqhi.py
#
# Full-formula AQHI for every PurpleAir sensor on the maps (added 2026-09-30):
#   PM2.5 = the sensor's own 3 h mean (Supabase sensor_readings, map-clean values)
#   O3    = ECCC RDAQA 10 km analysis at the sensor, 3 h mean (local archive,
#           /opt/airquality/scripts/archive_rdaqa.sh)
#   NO2   = nearest official AB station within 25 km (3 h mean, last6h.csv),
#           otherwise RDAQA NO2 at the sensor
# Fallback when RDAQA is missing or stale: PM-only estimate + monthly seasonal
# offset (same table as dk_LIFX/update_light_pa.py).
#
# Why this rule: drafts/regional_gas_eaqhi validation. At 9 co-located
# PurpleAir/station pairs (29,348 h) PM + regional gases landed within +/-1 of
# the official AQHI 98.8% of hours vs 92.9% seasonal and 79.9% PM-only, and it
# held in July (94%) where seasonal collapses (73%). At McCauley (urban),
# borrowing nearby station NO2 beat any rural-background NO2 guard. RDAQA beat
# station IDW for O3/NO2 (48 h test, not independent - RDAQA assimilates the
# stations). Using RDAQA NO2 instead of a fixed rural background away from
# stations is the one untested substitution; it is needed for BC/SK/NWT.
#
# Adds fields to data/{AB,BC,NT}_PM25_map.json in place and writes
# data/SK_band_PM25_map.json (SK_datapull sensors within 150 km of AB):
#   aqhi_rg, aqhi_rg_raw, aqhi_method, pm25_3h, o3_ppb, no2_ppb, no2_source,
#   gas_hour_utc

import glob
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RDAQA_DIR = Path("/opt/airquality/data/rdaqa_archive")
SK_JSON = Path("/opt/airquality/github/SK_datapull/data/SK_PM25_map.json")

REGIONS = ["AB", "BC", "NT"]
SK_BAND_KM = 150            # match PA_border_pull.py BUFFER_KM
NO2_STATION_KM = 25.0
RDAQA_MAX_AGE_H = 6
GAS_HOURS = 3

# Copied from dk_LIFX/update_light_pa.py MONTHLY_AQHI_CORRECTION (fit
# 2026-09-02 on 8 co-located pairs) - keep the two in step.
MONTHLY_AQHI_CORRECTION = {
    1: 1.07, 2: 1.19, 3: 1.08, 4: 1.34, 5: 1.26, 6: 0.69,
    7: 0.58, 8: 0.54, 9: 0.27, 10: 0.65, 11: 0.57, 12: 0.65,
}


def aqhi_raw(o3_ppb, no2_ppb, pm25):
    return (1000.0 / 10.4) * (
        math.exp(0.000537 * o3_ppb)
        + math.exp(0.000871 * no2_ppb)
        + math.exp(0.000487 * pm25)
        - 3.0
    )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------- inputs ----------------

def load_region(region):
    path = DATA_DIR / f"{region}_PM25_map.json"
    if not path.exists():
        return path, []
    with open(path) as f:
        return path, json.load(f)


def load_sk_band():
    """SK_datapull covers all of SK; keep the 150 km band next to AB."""
    if not SK_JSON.exists():
        return []
    with open(SK_JSON) as f:
        feats = json.load(f).get("features", [])
    recs = []
    for ft in feats:
        p = ft.get("properties", {})
        lon, lat = ft["geometry"]["coordinates"][:2]
        pm = p.get("pm_corrected_clean") if p.get("use_for_map") else None
        recs.append({
            "sensor_index": p.get("sensor_index"),
            "name": p.get("name"),
            "latitude": lat,
            "longitude": lon,
            "pm_corr": pm,
            "humidity": p.get("humidity"),
            "last_seen": p.get("last_seen"),
            "quality_flag": p.get("quality_flag"),
            "use_for_map": p.get("use_for_map"),
        })
    if not recs:
        return []
    ab = gpd.read_file(DATA_DIR / "Alberta.shp").to_crs(epsg=3978).geometry.union_all()
    pts = gpd.GeoSeries(
        gpd.points_from_xy([r["longitude"] for r in recs], [r["latitude"] for r in recs]),
        crs=4326,
    ).to_crs(epsg=3978)
    keep = (pts.distance(ab) <= SK_BAND_KM * 1000) & ~pts.intersects(ab)
    return [r for r, k in zip(recs, keep) if k]


def fetch_pm_3h(sensor_ids):
    """Mean of the last GAS_HOURS hourly map-clean PM2.5 values per sensor."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key or not sensor_ids:
        return {}
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=GAS_HOURS - 1)
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows = []
    ids = sorted(set(int(s) for s in sensor_ids))
    for i in range(0, len(ids), 150):
        chunk = ",".join(map(str, ids[i:i + 150]))
        offset = 0
        while True:
            r = requests.get(
                f"{url}/rest/v1/sensor_readings",
                headers={**headers, "Range": f"{offset}-{offset + 999}"},
                params={
                    "select": "sensor_index,recorded_at,pm_corrected_clean,use_for_map",
                    "sensor_index": f"in.({chunk})",
                    "recorded_at": f"gte.{start.isoformat()}",
                },
                timeout=30,
            )
            r.raise_for_status()
            page = r.json()
            rows += page
            if len(page) < 1000:
                break
            offset += 1000
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df = df[df["use_for_map"] == True]
    df["pm"] = pd.to_numeric(df["pm_corrected_clean"], errors="coerce")
    return df.dropna(subset=["pm"]).groupby("sensor_index")["pm"].mean().to_dict()


def rdaqa_hours():
    """Newest GAS_HOURS hours that have both O3 and NO2, newest first."""
    stamps = sorted(
        {Path(f).stem.split("_")[1] for f in glob.glob(str(RDAQA_DIR / "*/*/O3_*.tif"))},
        reverse=True,
    )
    out = []
    for st in stamps:
        o3 = next(iter(glob.glob(str(RDAQA_DIR / f"*/*/O3_{st}.tif"))), None)
        no2 = next(iter(glob.glob(str(RDAQA_DIR / f"*/*/NO2_{st}.tif"))), None)
        if o3 and no2:
            out.append((st, o3, no2))
        if len(out) == GAS_HOURS:
            break
    return out


def sample(path, lons, lats):
    """Nearest-cell value (ppb) at each point; NaN outside this file's extent."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
        vals = np.full(len(lons), np.nan)
        vals[ok] = arr[rows[ok], cols[ok]] * 1e9
    vals[(vals < 0) | (vals > 1000)] = np.nan
    return vals


def station_no2_3h():
    """AB stations' latest 3 h NO2 mean (ppb) from last6h.csv."""
    path = DATA_DIR / "last6h.csv"
    if not path.exists():
        return pd.DataFrame(columns=["StationName", "lat", "lon", "no2"])
    df = pd.read_csv(path)
    df = df[df["ParameterName"].astype(str).str.strip().str.lower() == "nitrogen dioxide"].copy()
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], errors="coerce", utc=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["ReadingDate", "Value", "Latitude", "Longitude"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
    out = []
    for name, g in df.groupby("StationName"):
        g = g.sort_values("ReadingDate").tail(GAS_HOURS)
        if len(g) < GAS_HOURS or g["ReadingDate"].iloc[-1] < cutoff:
            continue
        out.append({"StationName": name, "lat": g["Latitude"].iloc[0],
                    "lon": g["Longitude"].iloc[0], "no2": g["Value"].mean() * 1000})
    return pd.DataFrame(out)


# ---------------- main ----------------

def main():
    region_data = {r: load_region(r) for r in REGIONS}
    sk_band = load_sk_band()

    all_recs = [rec for _, recs in region_data.values() for rec in recs] + sk_band
    if not all_recs:
        print("No PurpleAir records found; nothing to do.")
        return

    lons = np.array([float(r["longitude"]) for r in all_recs])
    lats = np.array([float(r["latitude"]) for r in all_recs])

    pm3 = fetch_pm_3h([r["sensor_index"] for r in all_recs])

    # RDAQA 3 h means at every sensor
    hours = rdaqa_hours()
    now = datetime.now(timezone.utc)
    o3 = no2 = None
    gas_hour = None
    if hours:
        newest = datetime.strptime(hours[0][0], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if (now - newest).total_seconds() / 3600 <= RDAQA_MAX_AGE_H:
            o3 = np.nanmean([sample(h[1], lons, lats) for h in hours], axis=0)
            no2 = np.nanmean([sample(h[2], lons, lats) for h in hours], axis=0)
            gas_hour = newest.isoformat()
        else:
            print(f"RDAQA newest hour {hours[0][0]} is stale; using seasonal fallback")
    else:
        print("No RDAQA files; using seasonal fallback")

    stations = station_no2_3h()
    month_offset = MONTHLY_AQHI_CORRECTION.get(now.month, 0.0)
    counts = {"regional_gas": 0, "pm_seasonal": 0, "no_pm": 0, "no2_station": 0}

    for i, rec in enumerate(all_recs):
        pm_now = rec.get("pm_corr")
        if pm_now is None or not np.isfinite(float(pm_now)):
            rec.update(aqhi_rg=None, aqhi_rg_raw=None, aqhi_method=None)
            counts["no_pm"] += 1
            continue
        pm = float(pm3.get(int(rec["sensor_index"]), pm_now))

        o3_i = o3[i] if o3 is not None else np.nan
        no2_i = no2[i] if no2 is not None else np.nan
        no2_src = "rdaqa"
        if not stations.empty:
            d = haversine_km(lats[i], lons[i], stations["lat"].values, stations["lon"].values)
            j = int(np.argmin(d))
            if d[j] <= NO2_STATION_KM:
                no2_i = float(stations["no2"].iloc[j])
                no2_src = f"station:{stations['StationName'].iloc[j]} ({d[j]:.1f} km)"
                counts["no2_station"] += 1

        if np.isfinite(o3_i) and np.isfinite(no2_i):
            raw = aqhi_raw(o3_i, no2_i, pm)
            method = "regional_gas"
        else:
            raw = math.floor(pm / 10) + 1 + month_offset
            method = "pm_seasonal"
            o3_i = no2_i = np.nan
            no2_src = None
        counts[method] += 1

        rec.update(
            aqhi_rg=max(1, int(round(raw))),
            aqhi_rg_raw=round(raw, 2),
            aqhi_method=method,
            pm25_3h=round(pm, 2),
            o3_ppb=round(float(o3_i), 1) if np.isfinite(o3_i) else None,
            no2_ppb=round(float(no2_i), 1) if np.isfinite(no2_i) else None,
            no2_source=no2_src,
            gas_hour_utc=gas_hour if method == "regional_gas" else None,
        )

    for region, (path, recs) in region_data.items():
        if recs:
            with open(path, "w") as f:
                json.dump(recs, f, indent=2)
    with open(DATA_DIR / "SK_band_PM25_map.json", "w") as f:
        json.dump(sk_band, f, indent=2)

    sizes = {r: len(v[1]) for r, v in region_data.items()}
    sizes["SK_band"] = len(sk_band)
    print(f"Sensors: {sizes}; RDAQA hours used: {[h[0] for h in hours] if o3 is not None else 'none'}")
    print(f"Methods: {counts}")


if __name__ == "__main__":
    sys.exit(main())
