import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# ACA PurpleAir map file (data/ACA_PM25_map.json), rebuilt 2026-10-27 to follow the AB rules (Kevin).
#
# It used to pull PurpleAir itself with an old rule ported from R: at |A-B| 50-500 it took the HIGHER
# channel (a channel faulting at 300 next to a good 5 showed 300), no channel_override, no ceiling. Now it
# takes the ACA sensors straight from data/AB_PM25_map.json, which AB_PA_latest.py builds every 30 min with
# the agreed rules: per-channel 10-min averages with the fault guard, lower channel at (|A-B| > 10 and > 61%)
# or > 50, channel_override.csv, dead_list.csv, the same RH correction and the QA ceilings. One set of rules,
# one PurpleAir call. Sensors the AB pipeline marks use_for_map=false are left out, as before (the old file
# only listed sensors with a usable reading).
#
# Output format is unchanged (sensor_index, name, latitude, longitude, humidity, pm_corr, color, last_seen),
# and so is ACA's styling: sensors without "ACA" in the name are drawn grey.
# Side effect of the switch: indoor sensors drop out (the AB list is outdoor only) - e.g. 279217, now
# "MacEwan University 1", indoors at 82 F / 13% RH, was being shown as outdoor air.

AB_MAP = "data/AB_PM25_map.json"
MAX_AGE_MIN = 90            # refuse to republish if the AB file hasn't been rebuilt for this long


def get_color(pm, name):
    if "ACA" not in str(name):
        return "#808080"  # gray for non-ACA sensors
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


age_min = (datetime.now(timezone.utc).timestamp() - os.path.getmtime(AB_MAP)) / 60
if age_min > MAX_AGE_MIN:
    sys.exit(f"ERROR: {AB_MAP} is {age_min:.0f} min old - not republishing stale ACA data")

aca = pd.read_csv("data/ACA_sensors.csv")
aca_ids = set(aca["sensor_index"].dropna().astype(int))
ab = pd.DataFrame(json.load(open(AB_MAP)))
df = ab[ab["sensor_index"].astype(int).isin(aca_ids) & (ab["use_for_map"] != False) & ab["pm_corr"].notna()].copy()

df["color"] = df.apply(lambda x: get_color(x["pm_corr"], x["name"]), axis=1)
result = df[["sensor_index", "name", "latitude", "longitude", "humidity", "pm_corr", "color", "last_seen"]]
result.to_json("data/ACA_PM25_map.json", orient="records", indent=2)
print(f"ACA map: {len(result)} sensors (of {len(aca_ids)} on the ACA list) from {AB_MAP}, {age_min:.0f} min old")
