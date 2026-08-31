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
    

# approximate geographic bounds of the grid
lon_min, lon_max = -145, -85
lat_min, lat_max = 35, 75

lon_step = (lon_max - lon_min) / cols
lat_step = (lat_max - lat_min) / rows



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

for t in png_hours:

    print(f"Building PNG frame: {t:02d}h")

    if t >= pm.shape[0]:
        print(f"Skipping PNG {t:02d}h")
        continue

    if "LAY" in pm.dims:
        grid = pm.isel(TSTEP=t, LAY=0).values
    else:
        grid = pm.isel(TSTEP=t).values

    grid[grid < 0.1] = np.nan

    grid_ds = grid[::STEP, ::STEP]

    region_mask_ds = region_mask[::STEP, ::STEP]

    masked_grid = np.where(region_mask_ds, grid_ds, np.nan)

    # Find rows/columns containing valid smoke
    valid = ~np.isnan(masked_grid)
    
    if not np.any(valid):
        print(f"No valid smoke for {t:02d}h")
        continue
    
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    
    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    
    # Crop image
    masked_grid = masked_grid[rmin:rmax+1, cmin:cmax+1]
    
    # Geographic bounds of cropped image
    crop_lat_min = lat_ds[rmin]
    crop_lat_max = lat_ds[rmax] + lat_step * STEP
    crop_lon_min = lon_ds[cmin]
    crop_lon_max = lon_ds[cmax] + lon_step * STEP
    
    print(
        f"PNG bounds: "
        f"{crop_lat_min:.4f}, {crop_lon_min:.4f} -> "
        f"{crop_lat_max:.4f}, {crop_lon_max:.4f}"
    )
    
    alpha_mask = ~np.isnan(masked_grid)
    
    safe_grid = np.nan_to_num(masked_grid, nan=0.0)

    rgba = cmap(norm(safe_grid))

    rgba[..., 3] = np.where(alpha_mask, rgba[..., 3], 0)

    img = (rgba * 255).astype(np.uint8)

    img = np.flipud(img)

    image = Image.fromarray(img, mode="RGBA")

    UPSCALE = 3

    image = image.resize(
        (image.width * UPSCALE, image.height * UPSCALE),
        resample=Image.BICUBIC
    )

    png_out = DATA_DIR / f"firesmoke_{t:02d}h.png"

    image.save(png_out)

    print("Saved PNG:", png_out)

