#!/usr/bin/env python3
# scripts/extract_station_smoke_forecast.py
#
# Point-value PM2.5 smoke forecast per station, extracted from the same
# BlueSky/firesmoke.ca gridded forecast fetch_firesmoke.py already
# downloads for the province-wide raster/geojson layers - no new
# external data source, no new cost, just a different read of data
# already being pulled every 2 hours. For each station in the network,
# finds which forecast grid cell contains it and reports that cell's
# PM2.5 value at each of the 4 horizons (now/6h/12h/24h), giving a
# "what will it look like here over the next day" answer instead of
# reading a province-wide raster by eye.
#
# Run right after fetch_firesmoke.py (same cron cycle, see
# run_firesmoke.sh) so it always reads that run's fresh geojson output.

import json
import os
import sys
from pathlib import Path

import requests
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

DATA_DIR = Path("data/output")
HORIZONS = ["now", "6h", "12h", "24h"]
OUT_PATH = DATA_DIR / "station_smoke_forecast.json"


def load_stations():
    base_url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    r = requests.get(
        f"{base_url}/rest/v1/stations",
        headers=headers,
        params={"select": "StationName,Latitude,Longitude"},
        timeout=30,
    )
    r.raise_for_status()
    return [
        s for s in r.json()
        if s.get("Latitude") is not None and s.get("Longitude") is not None
    ]


def load_grid(horizon):
    path = DATA_DIR / f"firesmoke_{horizon}.geojson"
    with open(path) as f:
        data = json.load(f)

    geoms = []
    values = []
    for feat in data["features"]:
        geoms.append(shape(feat["geometry"]))
        values.append(feat["properties"]["pm25"])

    return STRtree(geoms), values


def lookup(tree, values, lon, lat):
    # "intersects" not "contains" - STRtree's "contains" predicate missed
    # real matches in testing (likely a boundary-precision quirk between
    # grid cell edges and station coordinates), while "intersects" is
    # exactly the right test anyway for "which grid cell is this point
    # in," boundary-inclusive.
    pt = Point(lon, lat)
    idx = tree.query(pt, predicate="intersects")
    if len(idx) == 0:
        return None
    return round(float(values[idx[0]]), 2)


def main():
    stations = load_stations()
    print(f"Loaded {len(stations)} stations with coordinates.")

    grids = {}
    for h in HORIZONS:
        path = DATA_DIR / f"firesmoke_{h}.geojson"
        if not path.exists():
            print(f"Missing {path}, skipping horizon {h}.")
            continue
        tree, values = load_grid(h)
        grids[h] = (tree, values)
        print(f"Loaded grid for {h}: {len(values)} cells.")

    if not grids:
        print("No forecast grids available - nothing to extract.")
        sys.exit(1)

    result = {}
    for st in stations:
        name = st["StationName"]
        lat, lon = float(st["Latitude"]), float(st["Longitude"])
        forecast = {}
        for h, (tree, values) in grids.items():
            forecast[h] = lookup(tree, values, lon, lat)
        result[name] = forecast

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"stations": result}, f)

    print(f"Wrote smoke forecast for {len(result)} stations -> {OUT_PATH}")


if __name__ == "__main__":
    main()
