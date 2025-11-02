import os, time, json, requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STATIONS_ODATA_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/Stations"
    "?$select=Name,Latitude,Longitude&$top=1000"
)
MEAS_URL = "https://data.environment.alberta.ca/EdwServices/aqhi/odata/StationMeasurements"

AB_TZ = ZoneInfo("America/Edmonton")  # robust across DST
DEFAULT_TIMEOUT = 45
PER_STATION_TOP = 5000
SLEEP_BETWEEN = 0.20  # seconds to avoid throttling

# ---------- session with robust retries ----------
def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
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
    """GET with retries and explicit logging on failure/JSON errors."""
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        # Log 1 sample full URL for debugging
        if params and os.environ.get("AQHI_DEBUG_SAMPLE_URL") != "done":
            full = f"{url}?{urlencode(params)}"
            print(f"[pull] sample GET: {full}")
            os.environ["AQHI_DEBUG_SAMPLE_URL"] = "done"
        if r.status_code >= 400:
            raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r
    except (requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError) as e:
        raise RuntimeError(f"Network timeout/connection error: {e}") from e

def read_json_safe(resp, context=""):
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        text = resp.text[:400].replace("\n", " ")
        raise RuntimeError(f"JSON decode failed {context}: {e}; first 400 chars: {text}")

def fetch_station_list() -> pd.DataFrame:
    resp = http_get(STATIONS_ODATA_URL, timeout=30)
    raw = read_json_safe(resp, context="(stations)")
    vals = raw.get("value", [])
    if not vals:
        print("[pull] WARNING: Stations payload is empty.")
        return pd.DataFrame(columns=["Name","Latitude","Longitude"])
    df = pd.json_normalize(vals)
    df = df.rename(columns={"Name":"Name","Latitude":"Latitude","Longitude":"Longitude"})
    return df[["Name","Latitude","Longitude"]]

def format_datetimeoffset(dt: datetime) -> str:
    # OData datetimeoffset'YYYY-MM-DDTHH:MM:SS±HH:MM'
    s = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return f"datetimeoffset'{s[:-2]}:{s[-2:]}'"

def fetch_page(url, params):
    """Fetch one page and return (rows, next_url_or_None)."""
    resp = http_get(url, params=params, timeout=DEFAULT_TIMEOUT)
    raw = read_json_safe(resp, context="(measurements)")
    rows = raw.get("value", [])
    next_link = raw.get("@odata.nextLink")
    return rows, next_link

def fetch_last6h_for_station(station_name: str) -> pd.DataFrame:
    now_ab = datetime.now(AB_TZ)
    start_ab = now_ab - timedelta(hours=6)
    start_literal = format_datetimeoffset(start_ab)
    safe_name = station_name.replace("'", "''")

    params = {
        "$format": "json",
        "$filter": f"StationName eq '{safe_name}' AND ReadingDate gt {start_literal}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
        "$top": str(PER_STATION_TOP),
    }

    rows, next_link = fetch_page(MEAS_URL, params)
    all_rows = list(rows)
    # If the service ever returns more than $top, follow nextLink:
    while next_link:
        # nextLink already contains encoded params
        resp = http_get(next_link, timeout=DEFAULT_TIMEOUT)
        raw = read_json_safe(resp, context="(measurements nextLink)")
        all_rows.extend(raw.get("value", []))
        next_link = raw.get("@odata.nextLink")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # predictable dtypes
    return df.astype(
        {"StationName":"string","ParameterName":"string","ReadingDate":"string"},
        errors="ignore"
    ).assign(Value=pd.to_numeric(pd.Series([r.get("Value") for r in all_rows]), errors="coerce"))

def probe_last6h_any() -> pd.DataFrame:
    """Broad probe: pull any measurements in last 6h (no StationName filter)."""
    now_ab = datetime.now(AB_TZ)
    start_ab = now_ab - timedelta(hours=6)
    start_literal = format_datetimeoffset(start_ab)
    params = {
        "$format": "json",
        "$filter": f"ReadingDate gt {start_literal}",
        "$orderby": "ReadingDate desc",
        "$select": "StationName,ParameterName,ReadingDate,Value",
        "$top": "2000",
    }
    rows, next_link = fetch_page(MEAS_URL, params)
    df = pd.DataFrame(rows)
    print(f"[probe] any-in-6h rows: {len(df)}")
    if not df.empty:
        print("[probe] sample:", df.head(5).to_string(index=False))
        # also show top parameter names to confirm AQHI label:
        if "ParameterName" in df:
            print("[probe] top ParameterName:", df["ParameterName"].value_counts().head(10).to_dict())
    return df

if __name__ == "__main__":
    stations_df = fetch_station_list()
    print(f">>> Fetched {len(stations_df)} stations.")

    combined_rows = []
    total_req = 0

    for _, row in stations_df.iterrows():
        name, lat, lon = row["Name"], row["Latitude"], row["Longitude"]
        df = fetch_last6h_for_station(name)
        total_req += 1
        if not df.empty:
            df["Latitude"]  = float(lat)
            df["Longitude"] = float(lon)
            combined_rows.append(df)
            print(f">>> Pulled {len(df)} rows for {name!r}.")
        else:
            print(f">>> No data in last 6h for {name!r}.")
        time.sleep(SLEEP_BETWEEN)

    usable = [d for d in combined_rows if not d.empty]
    if usable:
        combined_df = pd.concat(usable, ignore_index=True)
        combined_df = combined_df[["Value","StationName","ParameterName","ReadingDate","Latitude","Longitude"]]
    else:
        combined_df = pd.DataFrame(columns=["Value","StationName","ParameterName","ReadingDate","Latitude","Longitude"])
        # If absolutely nothing came back, run a one-off broad probe and log it:
        print("[pull] WARNING: No rows for any station. Running broad probe without StationName filter…")
        probe_last6h_any()

    cwd = os.getcwd()
    print(f">>> CWD: {cwd}")
    print(f">>> Total rows combined: {len(combined_df)}")
    if not combined_df.empty:
        print(">>> Sample rows:")
        print(combined_df.head().to_string(index=False))

    # Write CSV (even if empty, for downstream scripts to handle gracefully)
    out_dir = Path("data"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "last6h.csv"
    combined_df.to_csv(out_path, index=False)
    print(f">>> Wrote {len(combined_df)} rows to {out_path}")

    # List directory
    for p in sorted(out_dir.iterdir(), key=lambda x: x.name.lower()):
        print("   ", p.name)
