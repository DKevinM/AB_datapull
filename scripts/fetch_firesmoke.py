import requests
import xarray as xr
import numpy as np
import json
from PIL import Image
import matplotlib.colors as mcolors
from pathlib import Path
import urllib3
from datetime import datetime, timedelta


urllib3.disable_warnings()

DATA_DIR = Path("data/output")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# region of interest (Prairies / Saskatchewan focus)
LAT_MIN = 42   # down into northern US
LAT_MAX = 65   # mid NWT
LON_MIN = -130 # BC coast-ish
LON_MAX = -90  # Manitoba east

url = "https://services.firesmoke.ca/forecasts/current/dispersion.nc"
nc_file = DATA_DIR / "firesmoke.nc"

print("Downloading FireSmoke forecast...")

r = requests.get(url, verify=False, timeout=120)

with open(nc_file, "wb") as f:
    f.write(r.content)

print("Saved:", nc_file)

ds = xr.open_dataset(nc_file)
print(ds)
print("VARS:", list(ds.data_vars))

if "PM25" not in ds.data_vars:
    print(f"ERROR: PM25 not found in dataset. Available: {list(ds.data_vars)}")
    exit(1)

tflag = ds["TFLAG"].values
date = int(tflag[0, 0, 0])
time = int(tflag[0, 0, 1])
year = date // 1000
day = date % 1000
hour = time // 10000
minute = (time % 10000) // 100
second = time % 100
smoke_time = datetime(year, 1, 1) + timedelta(days=day-1)
smoke_time = smoke_time.replace(hour=hour, minute=minute, second=second)
print("FireSmoke timestamp:", smoke_time)

pm = ds["PM25"]

# Downsample step
STEP = 1

forecast_hours = {
    "now":0,
    "6h":6,
    "12h":12,
    "24h":24
}

pm = ds["PM25"]
print("PM25 dims:", pm.dims)
print("PM25 shape:", pm.shape)

# grid size
if "LAY" in pm.dims:
    rows = pm.shape[2]
    cols = pm.shape[3]
else:
    rows = pm.shape[1]
    cols = pm.shape[2]
    

# Grid geometry comes from the file itself (IOAPI attributes). Until 2026-10-14
# this was a hard-coded "approximate" box (-145..-85, 35..75) that put every
# smoke cell 450-1000 km from its true location.
if int(ds.attrs.get("GDTYP", 1)) != 1:
    raise SystemExit(f"Unexpected FireSmoke grid type GDTYP={ds.attrs.get('GDTYP')} (expected 1 = lat/lon)")
lon_min = float(ds.attrs["XORIG"])
lat_min = float(ds.attrs["YORIG"])
lon_step = float(ds.attrs["XCELL"])
lat_step = float(ds.attrs["YCELL"])
assert int(ds.attrs["NCOLS"]) == cols and int(ds.attrs["NROWS"]) == rows, "grid attrs disagree with PM25 shape"
lon_max = lon_min + lon_step * cols
lat_max = lat_min + lat_step * rows
print(f"FireSmoke grid: lon {lon_min}..{lon_max:.2f}, lat {lat_min}..{lat_max:.2f}, {lon_step} x {lat_step} deg")

# Forecast frames are relative to NOW, not to the model start (which can be
# several hours old): "now" = the model hour nearest the current clock.
_now_utc = datetime.utcnow()
hour0 = max(0, int(round((_now_utc - smoke_time).total_seconds() / 3600)))
print(f"Model start {smoke_time} UTC, now {_now_utc:%Y-%m-%d %H:%M} UTC -> 'now' = model hour {hour0}")



# Precompute lat/lon centers once
lat_vals = lat_min + np.arange(rows) * lat_step
lon_vals = lon_min + np.arange(cols) * lon_step

# Region mask once
region_mask = (
    (lat_vals[:, None] >= LAT_MIN) &
    (lat_vals[:, None] <= LAT_MAX) &
    (lon_vals[None, :] >= LON_MIN) &
    (lon_vals[None, :] <= LON_MAX)
)


colors = [
    (210/255,255/255,210/255,0.70),
    (180/255,255/255,180/255,0.78),
    (255/255,255/255,120/255,0.84),
    (255/255,200/255,80/255,0.88),
    (255/255,120/255,60/255,0.92),
    (220/255,60/255,40/255,0.96),
    (160/255,0,0,1.00)
]

cmap = mcolors.LinearSegmentedColormap.from_list(
    "smoke",
    colors
)

# Expands the colour range for low concentrations while retaining
# differentiation at higher smoke concentrations
norm = mcolors.PowerNorm(
    gamma=0.30,
    vmin=0.1,
    vmax=80
)



for name, t in forecast_hours.items():

    t = hour0 + t
    forecast_time = smoke_time + timedelta(hours=t)
    print("Processing:", name, forecast_time)

    if t >= pm.shape[0]:
        print(f"Skipping {name} — not available")
        continue
    
    if "LAY" in pm.dims:
        grid = pm.isel(TSTEP=t, LAY=0).values
    else:
        grid = pm.isel(TSTEP=t).values


    
    print(f"Grid shape: {grid.shape}, min: {np.nanmin(grid)}, max: {np.nanmax(grid)}")
    grid[grid < 0.1] = np.nan
    print(f"After filtering < 0.1: {np.count_nonzero(~np.isnan(grid))} valid cells")


    grid_ds = grid[::STEP, ::STEP]
    lat_ds = lat_vals[::STEP]
    lon_ds = lon_vals[::STEP]
    region_mask_ds = region_mask[::STEP, ::STEP]

    valid_mask = (~np.isnan(grid_ds)) & region_mask_ds
    valid_rc = np.argwhere(valid_mask)

    features = []

    for r_idx, c_idx in valid_rc:
        raw_val = float(grid_ds[r_idx, c_idx])
        lat = lat_ds[r_idx]
        lon = lon_ds[c_idx]
    
        poly = [
            [lon, lat],
            [lon + lon_step * STEP * 1.0, lat],
            [lon + lon_step * STEP * 1.0, lat + lat_step * STEP * 1.0],
            [lon, lat + lat_step * STEP * 1.0],
            [lon, lat]
        ]
    
        features.append({
            "type": "Feature",
            "properties": {
                "pm25": raw_val,
                "forecast": name,
                "timestamp": forecast_time.isoformat()
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly]
            }
        })

    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    outfile = DATA_DIR / f"firesmoke_{name}.geojson"

    with open(outfile, "w") as f:
        json.dump(geojson, f)

    print("Saved:", outfile, "features:", len(features), "time:", forecast_time)
    print(f"Region mask: {np.sum(region_mask_ds)} cells in region")
    print(f"Valid cells in region: {np.sum(valid_mask)} cells")




# ===============================
# PNG animation frames
# ===============================

png_hours = list(range(0, 25, 2))

# The maps (LiveMap, SK_Air_Map) draw every frame with L.imageOverlay over the
# fixed box [LAT_MIN, LON_MIN]..[LAT_MAX, LON_MAX]. Leaflet stretches an image
# linearly in Web Mercator, so each output row is sampled at the latitude that
# Mercator row really represents. (Before 2026-10-14 each frame was cropped to
# its own smoke extent and equirectangular, so it landed somewhere else again.)
PNG_W = 1200
_merc = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
_y_top, _y_bot = _merc(LAT_MAX), _merc(LAT_MIN)
PNG_H = int(round(PNG_W * (_y_top - _y_bot) / np.radians(LON_MAX - LON_MIN)))
_px_lon = LON_MIN + (np.arange(PNG_W) + 0.5) / PNG_W * (LON_MAX - LON_MIN)
_px_lat = np.degrees(2 * np.arctan(np.exp(_y_top - (np.arange(PNG_H) + 0.5) / PNG_H * (_y_top - _y_bot))) - np.pi / 2)
_ci = np.floor((_px_lon - lon_min) / lon_step).astype(int)
_rj = np.floor((_px_lat - lat_min) / lat_step).astype(int)       # IOAPI row 0 = southern edge
_ok = (_rj[:, None] >= 0) & (_rj[:, None] < rows) & (_ci[None, :] >= 0) & (_ci[None, :] < cols)
_ci, _rj = np.clip(_ci, 0, cols - 1), np.clip(_rj, 0, rows - 1)

for t in png_hours:

    print(f"Building PNG frame: {t:02d}h")

    k = hour0 + t
    if k >= pm.shape[0]:
        print(f"Skipping PNG {t:02d}h")
        continue

    if "LAY" in pm.dims:
        grid = pm.isel(TSTEP=k, LAY=0).values
    else:
        grid = pm.isel(TSTEP=k).values

    sampled = np.where(_ok, grid[_rj[:, None], _ci[None, :]], np.nan)
    sampled[sampled < 0.1] = np.nan

    alpha_mask = ~np.isnan(sampled)
    rgba = cmap(norm(np.nan_to_num(sampled, nan=0.0)))
    rgba[..., 3] = np.where(alpha_mask, rgba[..., 3], 0)
    image = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")   # row 0 = north already

    png_out = DATA_DIR / f"firesmoke_{t:02d}h.png"

    image.save(png_out)

    print("Saved PNG:", png_out)

