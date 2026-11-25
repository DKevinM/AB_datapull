"""
E-BAM 7-day rule (Kevin 2026-11-25): an E-BAM (portable PM2.5 monitor, names start "E-BAM") that has not
reported a real value for over 7 days comes off all our maps. Alberta keeps sending blank hourly rows for
dead E-BAMs (E-BAM Westview Village: last PM2.5 2026-11-10, still blank rows on 11-25), and every map read
those rows as "reporting" and drew an empty grey dot.

fetch_last6h.py calls update() on each new pull: it records each E-BAM's last real (non-blank) reading in
STATE_PATH (on dklinux, outside the repo), then returns the names to drop from last6h.csv. Dropping them
loses no data - an E-BAM silent for 7 days has only blank rows in a 24 h file. Every map and report that reads
last6h.csv (LiveMap, sit-reps, emergency maps, AQHI forecast, odour, SK map) then leaves it out on its own.
scripts/build_eAQHI.py reads off_map() so it doesn't put a PurpleAir estimate on the map in its place.

An E-BAM never seen with a value starts its clock the first time it is seen blank ("first_blank_utc"); the
initial history came from aqhi_data (seed run 2026-11-25). Other stations are not touched.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

STATE_PATH = Path(os.getenv("EBAM_STATE_PATH", "/opt/airquality/data/ebam_status.json"))
OFF_AFTER = timedelta(days=7)


def is_ebam(name):
    n = str(name or "").strip().upper().replace(" ", "")
    return n.startswith("E-BAM") or n.startswith("EBAM")


def _load():
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"stations": {}}


def _save(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(STATE_PATH)


def _is_off(rec, now):
    t = rec.get("last_value_utc") or rec.get("first_blank_utc")
    return bool(t) and now - pd.Timestamp(t).to_pydatetime() > OFF_AFTER


def update(df, now=None):
    """df: the combined last6h rows (Value, StationName, ParameterName, ReadingDate). Returns the set of
    E-BAM names to drop (silent over 7 days). Never raises - a failure here keeps every station."""
    try:
        now = now or datetime.now(timezone.utc)
        state = _load()
        st = state.setdefault("stations", {})
        d = df[df["StationName"].map(is_ebam)].copy()
        d["t"] = pd.to_datetime(d["ReadingDate"], errors="coerce", utc=True)
        d["v"] = pd.to_numeric(d["Value"], errors="coerce")
        for name, g in d.groupby("StationName"):
            rec = st.setdefault(name, {})
            good = g[g["v"].notna() & g["t"].notna()]
            if not good.empty:
                t = good["t"].max().isoformat()
                if not rec.get("last_value_utc") or t > rec["last_value_utc"]:
                    rec["last_value_utc"] = t
                rec.pop("first_blank_utc", None)
            elif not rec.get("last_value_utc") and not rec.get("first_blank_utc"):
                rec["first_blank_utc"] = now.isoformat()
            rec["last_row_utc"] = g["t"].max().isoformat() if g["t"].notna().any() else rec.get("last_row_utc")
        off = {n for n, r in st.items() if _is_off(r, now)}
        for n, r in st.items():
            r["on_map"] = n not in off
        state["updated_utc"] = now.isoformat()
        state["rule"] = "E-BAM with no real reading for over 7 days is left off all maps (Kevin 2026-11-25)"
        _save(state)
        return off
    except Exception as e:
        print(f"[ebam_status] skipped ({type(e).__name__}: {e}) - keeping every station")
        return set()


def off_map(now=None):
    """Names of E-BAMs currently off the maps (empty set if the state file is missing)."""
    now = now or datetime.now(timezone.utc)
    return {n for n, r in _load().get("stations", {}).items() if _is_off(r, now)}
