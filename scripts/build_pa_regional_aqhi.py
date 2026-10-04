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
# Local PM2.5 override (added 2026-10-04): the formula alone under-calls in
# smoke, because Alberta's published AQHI switches to a PM2.5 override then.
# drafts/regional_gas_eaqhi/test6b_results.txt, 7 co-located sites, official
# AQHI >= 7 (130 h): formula 26.9% within +/-1, PM-only 59.2%. So aqhi_rg is
# max(formula, PM-only floor(pm/10)+1) - but only after the local PM reading
# passes a sanity check (see question_pm_override), so one faulty sensor (the
# Jasper 2026-09-21 spike kind) can't paint a dot red on its own:
#   - sensors within NEIGHBOUR_KM: at least one must also read high
#   - no sensors nearby: a sudden jump must persist one full hourly reading
# A questioned override is not applied (the formula value stands), is marked
# aqhi_override="questioned" with the reason in aqhi_check, and newly
# questioned sensors are written to sitrep_alerts.log.
#
# Adds fields to data/{AB,BC,NT}_PM25_map.json in place and writes
# data/SK_band_PM25_map.json (SK_datapull sensors within 150 km of AB):
#   aqhi_rg, aqhi_rg_raw, aqhi_method, pm25_3h, o3_ppb, no2_ppb, no2_source,
#   gas_hour_utc, aqhi_pm_only, aqhi_override, aqhi_check

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

# PM override sanity check (question_pm_override)
QUESTION_MIN_PM = 30.0       # below this the override is at most AQHI 3 - not worth questioning
NEIGHBOUR_KM = 30.0
NEIGHBOUR_CONFIRM_FRAC = 0.3  # a neighbour confirms if it reads >= 30% of this sensor...
NEIGHBOUR_CONFIRM_MIN = 20.0  # ...and at least 20 ug/m3
JUMP_FACTOR = 3.0             # "sudden jump" = now > 3 x last hourly reading + 20
JUMP_ADD = 20.0
HIST_HOURS = 6
ALERTS_LOG = Path("/opt/airquality/logs/sitrep_alerts.log")
QUESTION_STATE = Path("/opt/airquality/logs/pa_override_questioned_state.json")

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


def fetch_pm_hist(sensor_ids):
    """Last HIST_HOURS hourly map-clean PM2.5 values per sensor, as
    {sensor_index: pd.Series indexed by hour, oldest first}."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key or not sensor_ids:
        return {}
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=HIST_HOURS - 1)
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
    df["t"] = pd.to_datetime(df["recorded_at"], utc=True)
    df = df.dropna(subset=["pm"]).sort_values("t")
    return {int(k): g.set_index("t")["pm"] for k, g in df.groupby("sensor_index")}


def pm_3h_means(hist):
    """Mean of the last GAS_HOURS hourly values per sensor (same window as before)."""
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=GAS_HOURS - 1)
    out = {}
    for k, ser in hist.items():
        recent = ser[ser.index >= start]
        if len(recent):
            out[k] = float(recent.mean())
    return out


def question_pm_override(i, pm_now, recs, lats, lons, live_pm, hist):
    """Return None if the local PM reading can be trusted to override the
    formula, else a short reason it is questioned."""
    if pm_now < QUESTION_MIN_PM:
        return None
    d = haversine_km(lats[i], lons[i], lats, lons)
    nb = (d <= NEIGHBOUR_KM) & np.isfinite(live_pm)
    nb[i] = False
    if nb.any():
        need = max(NEIGHBOUR_CONFIRM_FRAC * pm_now, NEIGHBOUR_CONFIRM_MIN)
        if (live_pm[nb] >= need).any():
            return None
        return (f"PM {pm_now:.0f} but {int(nb.sum())} sensor(s) within {NEIGHBOUR_KM:.0f} km "
                f"read at most {np.nanmax(live_pm[nb]):.0f}")
    ser = hist.get(int(recs[i]["sensor_index"]))
    if ser is None or ser.empty:
        return f"PM {pm_now:.0f}, no nearby sensor and no recent history to confirm it"
    last = float(ser.iloc[-1])
    if pm_now > JUMP_FACTOR * last + JUMP_ADD:
        return (f"PM jumped to {pm_now:.0f} from {last:.0f} at the last hourly reading, "
                f"no sensor within {NEIGHBOUR_KM:.0f} km to confirm - held until it persists")
    return None


def alert_new_questions(questioned):
    """Write newly questioned sensors (and ones that cleared) to sitrep_alerts.log."""
    try:
        prev = set(json.loads(QUESTION_STATE.read_text()))
    except Exception:
        prev = set()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"{now} ALERT pa_pm_override: sensor {sid} ({name}) questioned - {why}; "
             f"PM override NOT applied, map shows the formula value"
             for sid, (name, why) in questioned.items() if str(sid) not in prev]
    cleared = sorted(prev - {str(s) for s in questioned})
    if cleared:
        lines.append(f"{now} ALERT pa_pm_override: resolved - no longer questioned: {', '.join(cleared)}")
    if lines:
        with open(ALERTS_LOG, "a") as f:
            f.write("\n".join(lines) + "\n")
    QUESTION_STATE.write_text(json.dumps(sorted(str(s) for s in questioned)))


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

    hist = fetch_pm_hist([r["sensor_index"] for r in all_recs])
    pm3 = pm_3h_means(hist)
    live_pm = np.array([
        float(r["pm_corr"]) if r.get("pm_corr") is not None and r.get("use_for_map") is not False
        and np.isfinite(float(r["pm_corr"])) else np.nan
        for r in all_recs
    ])
    questioned = {}

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
    counts = {"regional_gas": 0, "pm_seasonal": 0, "no_pm": 0, "no2_station": 0,
              "pm_override": 0, "pm_override_questioned": 0}

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

        pm_only = math.floor(float(pm_now) / 10) + 1
        override = check = None
        if np.isfinite(o3_i) and np.isfinite(no2_i):
            raw = aqhi_raw(o3_i, no2_i, pm)
            method = "regional_gas"
            if pm_only > round(raw):
                check = question_pm_override(i, float(pm_now), all_recs, lats, lons, live_pm, hist)
                if check is None:
                    raw = float(pm_only)
                    override = "applied"
                    counts["pm_override"] += 1
                else:
                    override = "questioned"
                    counts["pm_override_questioned"] += 1
                    questioned[int(rec["sensor_index"])] = (rec.get("name"), check)
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
            aqhi_pm_only=pm_only,
            aqhi_override=override,
            aqhi_check=check,
        )

    try:
        alert_new_questions(questioned)
    except Exception as e:
        print(f"WARN: could not write override alerts: {e}")
    for sid, (name, why) in questioned.items():
        print(f"Questioned PM override: {sid} ({name}): {why}")

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
