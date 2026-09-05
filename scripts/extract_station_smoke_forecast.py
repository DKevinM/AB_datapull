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

import csv
import json
import sys
from pathlib import Path

from shapely.geometry import shape, Point
from shapely.strtree import STRtree

DATA_DIR = Path("data/output")
LAST6H_PATH = Path("data/last6h.csv")
HORIZONS = ["now", "6h", "12h", "24h"]
OUT_PATH = DATA_DIR / "station_smoke_forecast.json"


def load_stations():
    # Reads the same last6h.csv LiveMap itself reads for its station
    # list (js/data.js), not the separate Supabase `stations` table -
    # that table is populated from Alberta's official Stations OData
    # endpoint and doesn't include every station last6h.csv does (e.g.
    # "Olds Sensor", "Bremner" - community/ancillary sensors, not part
    # of the official registry). Sourcing from last6h.csv guarantees a
    # forecast entry for every station name LiveMap can actually show.
    seen = {}
    with open(LAST6H_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("StationName")
            lat, lon = row.get("Latitude"), row.get("Longitude")
            if not name or name in seen or not lat or not lon:
                continue
            try:
                seen[name] = {"StationName": name, "Latitude": float(lat), "Longitude": float(lon)}
            except ValueError:
                continue
    return list(seen.values())


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
    #
    # A miss here isn't "unknown" - fetch_firesmoke.py drops any cell
    # below 0.1 ug/m3 before writing the geojson (keeps the 100k+-feature
    # file from being even bigger with meaningless near-zero background),
    # so no matching cell means genuinely clean air, not missing data.
    # All AB/SK stations sit well inside the smoke model's domain, so
    # that's the only reason a miss happens here.
    pt = Point(lon, lat)
    idx = tree.query(pt, predicate="intersects")
    if len(idx) == 0:
        return "<0.1"
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
