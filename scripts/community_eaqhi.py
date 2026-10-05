#!/usr/bin/env python3
# scripts/community_eaqhi.py
#
# One estimated AQHI (eAQHI) per town from that town's PurpleAir sensors
# (added 2026-10-05). Called by build_pa_regional_aqhi.py once every sensor
# has its pm_corr / pm25_3h / O3 / NO2 / override fields. Writes
# data/community_eaqhi.json, the ONE place this is calculated: the LiveMap
# town diamonds, the LIFX bulbs (/opt/airquality/lifx_lights, source
# pa_community) and the dk_LIFX status page all read that file.
#
# Rule = the LiveMap dot rule applied to the town average:
#   max(AQHI formula on the town's 3 h PM2.5 + regional O3/NO2,
#       PM-only floor(PM2.5 now / 10) + 1)
# Fault handling before averaging:
#   3+ sensors: drop any reading more than max(30, 3 x median) from the
#               median (dk_LIFX / Pembina paper Section 3.5 rule)
#   2 sensors:  if they differ that much, use the LOWER one and flag it
#               (Kevin 2026-10-05: a lone spike 0.5 km from a clean
#               neighbour is almost never real - the Jasper 2026-09-21 case)
#   1 sensor:   if the per-sensor check questioned its PM override, no
#               override here either

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

TOWNS_FILE = Path(__file__).resolve().parent / "community_towns.json"
AB_TZ = ZoneInfo("America/Edmonton")   # last_seen in the map records is AB local time
OUTLIER_ABS = 30.0
OUTLIER_REL = 3.0


def aqhi_formula(o3_ppb, no2_ppb, pm25):
    """Stieb et al. (2008), corrected coefficients."""
    return (1000.0 / 10.4) * (math.exp(0.000537 * o3_ppb) + math.exp(0.000871 * no2_ppb)
                              + math.exp(0.000487 * pm25) - 3.0)


def _seen_utc(rec):
    try:
        return datetime.strptime(rec["last_seen"], "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=AB_TZ).astimezone(timezone.utc)
    except (TypeError, KeyError, ValueError):
        return None


def town_eaqhi(town, by_id):
    out = {"name": town["name"], "label": town.get("label", town["name"]),
           "lat": town["lat"], "lon": town["lon"],
           "sensors_configured": town["sensors"], "sensors_used": [],
           "eaqhi": None, "warning": None, "status": "no_data"}
    recs = []
    for sid in town["sensors"]:
        r = by_id.get(int(sid))
        try:
            pm = float(r["pm_corr"])
        except (TypeError, KeyError, ValueError):
            continue
        if r.get("use_for_map") is False or not math.isfinite(pm):
            continue
        recs.append(r)
    if not recs:
        out["how"] = "no fresh sensors"
        return out

    pms = [float(r["pm_corr"]) for r in recs]
    if len(recs) >= 3:
        med = median(pms)
        keep = [abs(p - med) <= max(OUTLIER_ABS, OUTLIER_REL * med) for p in pms]
        dropped = [f"{r['sensor_index']} ({p:.0f})" for r, p, k in zip(recs, pms, keep) if not k]
        if dropped:
            out["warning"] = f"sensor(s) {', '.join(dropped)} rejected vs town median {med:.0f}"
        recs = [r for r, k in zip(recs, keep) if k]
    elif len(recs) == 2:
        lo, hi = sorted(pms)
        if hi - lo > max(OUTLIER_ABS, OUTLIER_REL * lo):
            low = recs[pms.index(lo)]
            out["warning"] = (f"its 2 sensors disagree ({hi:.0f} vs {lo:.0f}) - using the lower "
                              f"({low['sensor_index']}); check the other one")
            recs = [low]

    pm_now = sum(float(r["pm_corr"]) for r in recs) / len(recs)
    pm3s = [float(r["pm25_3h"]) for r in recs if r.get("pm25_3h") is not None]
    pm3 = sum(pm3s) / len(pm3s) if pm3s else pm_now
    pm_only = math.floor(pm_now / 10) + 1
    gases = [(r["o3_ppb"], r["no2_ppb"]) for r in recs
             if r.get("aqhi_method") == "regional_gas" and r.get("o3_ppb") is not None and r.get("no2_ppb") is not None]
    if gases:
        o3 = sum(g[0] for g in gases) / len(gases)
        no2 = sum(g[1] for g in gases) / len(gases)
        formula = round(aqhi_formula(o3, no2, pm3))
        questioned = len(recs) == 1 and recs[0].get("aqhi_override") == "questioned"
        aqhi = formula if questioned else max(formula, pm_only)
        local_pm = (pm_only > formula) and not questioned
        out.update(method="regional_gas", o3_ppb=round(o3, 1), no2_ppb=round(no2, 1),
                   formula=formula, pm_only=pm_only, set_by_local_pm=local_pm,
                   how=(f"PurpleAir + regional O3 {o3:.0f} / NO2 {no2:.0f} ppb"
                        + (", set by local PM2.5" if local_pm else "")
                        + (" (high PM2.5 not confirmed, not used)" if questioned else "")))
    else:
        raws = [float(r["aqhi_rg_raw"]) for r in recs if r.get("aqhi_rg_raw") is not None]
        aqhi = round(sum(raws) / len(raws)) if raws else pm_only
        out.update(method="pm_seasonal", pm_only=pm_only, set_by_local_pm=False,
                   how="PurpleAir + seasonal adjustment (regional gases unavailable)")

    seen = [t for t in (_seen_utc(r) for r in recs) if t]
    out.update(eaqhi=min(max(int(aqhi), 1), 10), status="ok",
               pm25=round(pm_now, 1), pm25_3h=round(pm3, 1),
               sensors_used=[int(r["sensor_index"]) for r in recs],
               reading_time_utc=max(seen).isoformat(timespec="minutes") if seen else None)
    return out


def write_community_eaqhi(all_recs, out_path):
    towns = json.loads(TOWNS_FILE.read_text())["towns"]
    by_id = {int(r["sensor_index"]): r for r in all_recs if r.get("sensor_index") is not None}
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "max(AQHI formula on town 3h PM2.5 + regional O3/NO2, floor(PM2.5/10)+1); see scripts/community_eaqhi.py",
        "towns": [town_eaqhi(t, by_id) for t in towns],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result
