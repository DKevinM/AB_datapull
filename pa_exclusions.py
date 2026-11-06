# Areas whose PurpleAir sensors stay off the maps and the model, whatever their
# id. Added 2026-11-06, Kevin: Environment Canada keeps a rack of test units on
# its Edmonton building, and new batches arrive with new ids, so a dead list
# alone can't keep up. Bounds: every unit there, with about 50 m of margin.
import pandas as pd

TEST_BOXES = [
    # name, lat_min, lat_max, lon_min, lon_max
    ("ECCC Edmonton test rack", 53.5265, 53.5335, -113.4190, -113.4125),
]


def test_box_mask(df):
    """True for rows inside a test box. Blank coordinates are never inside."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    mask = pd.Series(False, index=df.index)
    for _, lat_min, lat_max, lon_min, lon_max in TEST_BOXES:
        mask |= lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
    return mask
