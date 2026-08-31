#!/usr/bin/env python3
# fetch_last6h.py — pulls last 24h of AQHI station measurements (name kept),
# writes data/last6h.csv. Robust to OData quirks, paging, and timeouts.

import os, time, json, requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------ Config (names unchanged) ------------------
STATIONS_ODATA_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    "?$select=Name,Latitude,Longitude&$top=1000"
)
MEAS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"

AB_TZ = ZoneInfo("America/Edmonton")     # robust across DST
LOOKBACK_HOURS = int(os.getenv("AQHI_LOOKBACK_HOURS", "24"))  # keep file name, widen window
DEFAULT_TIMEOUT = 45
PER_STATION_TOP = int(os.getenv("AQHI_PER_STATION_TOP", "10000"))
SLEEP_BETWEEN = float(os.getenv("AQHI_SLEEP_BETWEEN", "0.20"))  # throttle to avoid 429

# ------------------ HTTP session with retries ------------------
def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Accept": "application/json"})
    return s

SESSION = make_session()

def http_get(url, *, params=None, timeout=DEFAULT_TIMEOUT):
    """GET with retries; logs a sample full URL for debugging."""
    r = SESSION.get(url, params=params, timeout=timeout)
    if params and os.environ.get("AQHI_DEBUG_SAMPLE_URL") != "done":
        full = f"{url}?{urlencode(params)}"
        print(f"[pull] sample GET: {full}")
        os.environ["AQHI_DEBUG_SAMPLE_URL"] = "done"
    if r.status_code >= 400:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r

def read_json_safe(resp, context=""):
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        text = resp.text[:400].replace("\n", " ")
        raise RuntimeError(f"JSON decode failed {context}: {e}; first 400 chars: {text}")

# ------------------ Helpers ------------------
def iso_offset(dt: datetime) -> str:
    """YYYY-MM-DDTHH:MM:SS±HH:MM (no quotes/wrappers)."""
    return dt.astimezone().isoformat(timespec="seconds")

def iso_utc(dt: datetime) -> str:
    """YYYY-MM-DDTHH:MM:SSZ (UTC)."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

def build_filter(station_name: str, dt_literal: str) -> str:
    safe_name = station_name.replace("'", "''")
    return f"StationName eq '{safe_name}' AND ReadingDate gt {dt_literal}"

def fetch_page(url, params):
    """Fetch one OData page -> (rows, nextLink)."""
    resp = http_get(url, params=params, timeout=DEFAULT_TIMEOUT)
    raw = read_json_safe(resp, context="(page)")
    return raw.get("value", []), raw.get("@odata.nextLink")

# ------------------ API calls ------------------
def fetch_station_list() -> pd.DataFrame:
    resp = http_get(STATIONS_ODATA_URL, timeout=30)
    raw = read_json_safe(resp, context="(stations)")
    vals = raw.get("value", [])
    if not vals:
        print("[pull] WARNING: Stations payload is empty.")
        return pd.DataFrame(columns=["Name", "Latitude", "Longitude"])
    df = pd.json_normalize(vals)
    return df[["Name", "Latitude", "Longitude"]]

def try_fetch_station(meas_url, params_base, dt_literal_styles):
    """Try date literal styles until one yields non-400. Return (rows, next, style_used)."""
    last_err = None
    for style_name, dt_lit in dt_literal_styles:
        params = params_base.copy()
        params["$filter"] = build_filter(params_base["__station_name"], dt_lit)
        try:
            rows, next_link = fetch_page(meas_url, {k:v for k,v in params.items() if not k.startswith("__")})
            return rows, next_link, style_name
        except requests.HTTPError as e:
            if "HTTP 400" in str(e):
                print(f"[pull] 400 with style={style_name}; trying next…")
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("All date literal styles failed.")

def fetch_last6h_for_station(station_name: str) -> pd.DataFrame:
    # (name kept; actually pulls LOOKBACK_HOURS)
    now_ab = datetime.now(AB_TZ)
    start_ab = now_ab - timedelta(hours=LOOKBACK_HOURS)
    print(f"[pull] lookback={LOOKBACK_HOURS}h; window starts (America/Edmonton): {start_ab.isoformat(timespec='seconds')}")

    # Try these literal styles in order (no quotes first):
    dt_styles = [
        ("iso-offset", iso_offset(start_ab)),                      # 2025-11-01T21:34:20-06:00
        ("iso-utc-Z",  iso_utc(start_ab)),                         # 2025-11-02T03:34:20Z
        ("legacy-datetime", f"datetime'{iso_utc(start_ab)}'"),     # datetime'2025-11-02T03:34:20Z'
    ]

    params_base = {
        "__station_name": station_name,
        "$format": "json",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
        "$top": str(PER_STATION_TOP),
    }

    rows, next_link, style_used = try_fetch_station(MEAS_URL, params_base, dt_styles)

    # Follow paging if provided
    all_rows = list(rows)
    while next_link:
        resp = http_get(next_link, timeout=DEFAULT_TIMEOUT)
        raw = read_json_safe(resp, context="(measurements nextLink)")
        all_rows.extend(raw.get("value", []))
        next_link = raw.get("@odata.nextLink")

    if not all_rows:
        return pd.DataFrame()

    print(f"[pull] station={station_name!r} used date-style={style_used} rows={len(all_rows)}")
    df = pd.DataFrame(all_rows)
    # predictable dtypes
    if not df.empty:
        df = df.astype({"StationName": "string", "ParameterName": "string", "ReadingDate": "string"},
                       errors="ignore")
        df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce")
    return df

def probe_last6h_any() -> pd.DataFrame:
    """Broad probe without StationName, useful when everything is empty."""
    now_ab = datetime.now(AB_TZ)
    start_ab = now_ab - timedelta(hours=LOOKBACK_HOURS)
    styles = [("iso-offset", iso_offset(start_ab)), ("iso-utc-Z", iso_utc(start_ab))]
    for name, lit in styles:
        params = {
            "$format": "json",
            "$filter": f"ReadingDate gt {lit}",
            "$orderby": "ReadingDate desc",
            "$select": "StationName,ParameterName,ReadingDate,Value",
            "$top": "2000",
        }
        try:
            rows, _ = fetch_page(MEAS_URL, params)
            df = pd.DataFrame(rows)
            print(f"[probe] style={name} rows={len(df)}")
            if not df.empty:
                print("[probe] sample:\n", df.head(5).to_string(index=False))
                if "ParameterName" in df.columns:
                    print("[probe] top ParameterName:", df["ParameterName"].value_counts().head(10).to_dict())
                return df
        except Exception as e:
            print(f"[probe] style={name} failed: {e}")
    return pd.DataFrame()

# ------------------ Main ------------------
def already_current(out_path, now=None):
    """Checks the EXISTING last6h.csv (no network call) to decide whether
    this run needs to hit the government API at all.

    Government data is 'Hour Ending' - a reading timestamped e.g. 11:00
    covers 10:00-11:00 and is normally published somewhere between ~10 and
    ~35 min after the hour turns over (observed empirically 2026-08-31).
    Comparing against 'the next hour we're waiting for' (not just 'now
    floored to the hour') means this stays correct even if we're behind by
    more than one hour, not just the most recent one.

    Backoff: if the next-needed hour is still missing more than 60 min
    after it should have appeared, that's not normal lag anymore - it's a
    likely government-side outage. Falls back to the original twice-hourly
    cadence (:10/:30) instead of continuing to hit their API every run, so
    a real outage doesn't turn into hammering their server while it's down.

    Returns (skip: bool, reason: str).
    """
    now = now or datetime.now(AB_TZ)

    if not out_path.exists():
        return False, "no existing file - must fetch"

    try:
        existing = pd.read_csv(out_path)
        have_hour = pd.to_datetime(existing["ReadingDate"], errors="coerce", utc=True).max()
    except Exception as e:
        return False, f"couldn't read existing file ({e}) - must fetch"

    if pd.isna(have_hour):
        return False, "existing file has no valid ReadingDate - must fetch"

    have_hour = have_hour.tz_convert(AB_TZ)
    next_expected = have_hour + timedelta(hours=1)

    if now < next_expected:
        return True, f"already have the current hour ({have_hour.strftime('%H:%M')})"

    gap_minutes = (now - next_expected).total_seconds() / 60

    if gap_minutes <= 60:
        return False, f"missing {next_expected.strftime('%H:%M')}, {gap_minutes:.0f}min overdue - normal lag, fetching"

    # Backoff: only actually fetch on the original :10/:30 cadence once a
    # gap has run past an hour - anything else is a wasted hit during what
    # looks like a real outage.
    if now.minute in (10, 30) or (now.minute in (11, 31)):
        return False, f"missing {next_expected.strftime('%H:%M')}, {gap_minutes:.0f}min overdue (backoff mode) - trying on :10/:30 tick"
    return True, f"missing {next_expected.strftime('%H:%M')}, {gap_minutes:.0f}min overdue (backoff mode) - skipping this off-cycle tick"


if __name__ == "__main__":
    _out_path = Path("data") / "last6h.csv"
    _skip, _reason = already_current(_out_path)
    print(f"[precheck] {_reason}")
    if _skip:
        raise SystemExit(0)

    stations_df = fetch_station_list()
    print(f">>> Fetched {len(stations_df)} stations.")

    combined_rows = []
    for _, row in stations_df.iterrows():
        name, lat, lon = row["Name"], row["Latitude"], row["Longitude"]
        try:
            df = fetch_last6h_for_station(name)
        except Exception as e:
            print(f">>> Error fetching {name!r}: {e}")
            df = pd.DataFrame()

        if not df.empty:
            df["Latitude"]  = float(lat)
            df["Longitude"] = float(lon)
            combined_rows.append(df)
            print(f">>> Pulled {len(df)} rows for {name!r}.")
        else:
            print(f">>> No data in last {LOOKBACK_HOURS}h for {name!r}.")
        time.sleep(SLEEP_BETWEEN)

    usable = [d for d in combined_rows if not d.empty]
    if usable:
        combined_df = pd.concat(usable, ignore_index=True)
        combined_df = combined_df[["Value", "StationName", "ParameterName", "ReadingDate", "Latitude", "Longitude"]]
    else:
        combined_df = pd.DataFrame(columns=["Value", "StationName", "ParameterName", "ReadingDate", "Latitude", "Longitude"])
        print("[pull] WARNING: No rows for any station. Running broad probe without StationName filter…")
        probe_last6h_any()

    cwd = os.getcwd()
    print(f">>> CWD: {cwd}")
    print(f">>> Total rows combined: {len(combined_df)}")
    if not combined_df.empty:
        print(">>> Sample rows:\n", combined_df.head().to_string(index=False))

    # Write CSV (name kept the same)
    out_dir = Path("data"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "last6h.csv"
    combined_df.to_csv(out_path, index=False)
    print(f">>> Wrote {len(combined_df)} rows to {out_path}")




    # ------------------ WRITE GEOJSON ------------------
    geojson_path = out_dir / "stations.geojson"
    features = []
    if not combined_df.empty:
        combined_df["ReadingDate"] = pd.to_datetime(combined_df["ReadingDate"], errors="coerce")
        # ----------------------------
        # FIX 1 — APPLY UNIT CONVERSION
        # ----------------------------
        convert_params = [
            "Ozone","Total Oxides of Nitrogen","Hydrogen Sulphide",
            "Total Reduced Sulphur","Sulphur Dioxide",
            "Nitric Oxide","Nitrogen Dioxide"
        ]
        combined_df.loc[
            combined_df["ParameterName"].isin(convert_params),
            "Value"
        ] *= 1000
        # ----------------------------
        # FIX 2 — REMOVE NaN
        # ----------------------------
        combined_df["Value"] = pd.to_numeric(combined_df["Value"], errors="coerce")
        combined_df = combined_df.dropna(subset=["Value"])


        # ----------------------------
        # FIX 3 — GET LATEST PER PARAM
        # ----------------------------
        latest = (
            combined_df.sort_values("ReadingDate")
            .groupby(["StationName", "ParameterName"], as_index=False)
            .last()
        )
        # ----------------------------
        # BUILD FEATURES
        # ----------------------------
        for station, group in latest.groupby("StationName"):
            lat = float(group["Latitude"].iloc[0])
            lon = float(group["Longitude"].iloc[0])
            props = {
                "stationName": station
            }
            for _, row in group.iterrows():
                param = row["ParameterName"]
                val = float(row["Value"])
                props[param] = val

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": props
            })    
    

    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, allow_nan=False)  # CRITICAL
    
    print(f">>> Wrote {len(features)} features to {geojson_path}")


    
    # List directory contents
    for p in sorted(out_dir.iterdir(), key=lambda x: x.name.lower()):
        print("   ", p.name)
