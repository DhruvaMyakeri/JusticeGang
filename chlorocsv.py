# andaman_chl_extract.py
# Requirements:
# pip install xarray netcdf4 numpy matplotlib cartopy pandas rioxarray shapely geopandas

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Optional for shapefile overlay:
try:
    import geopandas as gpd
    has_gpd = True
except Exception:
    has_gpd = False

# --------- USER INPUT ---------
# Path to your MODIS file (NetCDF / xarray-readable)
MODIS_PATH = "D:\INFERENTIA\AQUA_MODIS.20250731.L3m.DAY.CHL.chlor_a.9km.nc"   # <- change to your filename

# Bounding box for Andaman region (degrees)
# Default chosen to cover Andaman & Nicobar reefs; adjust if needed.
lat_min, lat_max = 6.5, 14.5     # degrees North
lon_min, lon_max = 92.0, 94.5    # degrees East

# If you have a reef shapefile, put path here to overlay:
REEF_SHAPEFILE = None   # e.g. "andaman_reefs.shp"  (optional)

# Output filenames
OUT_DIR = "andaman_chl_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------- LOAD DATA ---------
ds = xr.open_dataset(MODIS_PATH)
print(ds)  # quick inspect

# Common variable name used in your file: 'chlor_a'
varname = "chlor_a"

if varname not in ds:
    raise ValueError(f"{varname} not found in dataset. Available data variables: {list(ds.data_vars)}")

chl = ds[varname]

# Ensure longitude range is -180..180 or 0..360 consistent with the file
lon = ds.coords.get("lon", ds.coords.get("longitude"))
lat = ds.coords.get("lat", ds.coords.get("latitude"))

# If lon runs 0..360 and you supplied 92..94 (ok), but if you have -180..180 it's also ok.
# We'll normalize both dataset and bbox to -180..180 for safe selection:
def normalize_lon(lon_arr):
    lon_arr = lon_arr % 360
    lon_arr = (lon_arr + 180) % 360 - 180
    return lon_arr

# If coords are 1D:
if 'lon' in ds.coords:
    ds = ds.assign_coords(lon=normalize_lon(ds.lon))
elif 'longitude' in ds.coords:
    ds = ds.assign_coords(longitude=normalize_lon(ds.longitude))

# Normalize requested bbox to -180..180 as well
def norm(b):
    return ((b + 180) % 360) - 180
lon_min_n, lon_max_n = norm(lon_min), norm(lon_max)

# If lon_min_n > lon_max_n (crosses dateline), handle separately.
crosses_dateline = lon_min_n > lon_max_n

# --------- SUBSET ---------
# Select lat slice
ds_sub_lat = ds.sel(lat=slice(lat_max, lat_min)) if ds.lat.values[0] > ds.lat.values[-1] else ds.sel(lat=slice(lat_min, lat_max))

# Select lon slice (handle dateline if needed)
if not crosses_dateline:
    ds_sub = ds_sub_lat.sel(lon=slice(lon_min_n, lon_max_n))
else:
    # join two slices
    left = ds_sub_lat.sel(lon=slice(lon_min_n, 180))
    right = ds_sub_lat.sel(lon=slice(-180, lon_max_n))
    ds_sub = xr.concat([left, right], dim="lon")

chl_sub = ds_sub[varname]

# --------- CLEAN / MASK ---------
# Typical MODIS fill values are NaN or extreme values; use dataset attribute if present:
fill_value = chl_sub.attrs.get("_FillValue", None) or chl_sub.encoding.get("_FillValue", None)
chl_clean = chl_sub.where(np.isfinite(chl_sub))  # remove NaN/inf
if fill_value is not None:
    chl_clean = chl_clean.where(chl_clean != fill_value)

# Often negative chlorophyll means invalid -> mask
chl_clean = chl_clean.where(chl_clean > 0)

# --------- BASIC STATS ---------
flat = chl_clean.values.flatten()
flat = flat[np.isfinite(flat)]
if flat.size == 0:
    raise ValueError("No valid chlorophyll values in the selected region. Try a larger bbox or inspect the file.")

mean_chl = float(np.nanmean(flat))
median_chl = float(np.nanmedian(flat))
p10, p90 = float(np.nanpercentile(flat, 10)), float(np.nanpercentile(flat, 90))
min_chl, max_chl = float(np.nanmin(flat)), float(np.nanmax(flat))

# Area-weighted mean (cosine latitude weights)
lats = chl_clean['lat'].values
# build 2D weights matching chl grid
weights_1d = np.cos(np.deg2rad(lats))
# expand dims if needed: assume chl is (lat, lon)
weights_2d = xr.DataArray(weights_1d, coords=[chl_clean.lat], dims=["lat"])
weighted_mean = float((chl_clean * weights_2d).sum(dim=("lat","lon")) / weights_2d.sum(dim="lat"))

# Save summary
summary = {
    "lat_range": [lat_min, lat_max],
    "lon_range": [lon_min, lon_max],
    "n_pixels": int(np.sum(np.isfinite(chl_clean.values))),
    "mean_chl": mean_chl,
    "median_chl": median_chl,
    "weighted_mean_chl": weighted_mean,
    "p10": p10,
    "p90": p90,
    "min": min_chl,
    "max": max_chl
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "andaman_chl_summary.csv"), index=False)
print("Summary saved to:", os.path.join(OUT_DIR, "andaman_chl_summary.csv"))
print(summary)

# --------- SAVE CSV of grid values (lat, lon, chl) ---------
df = chl_clean.to_dataframe(name="chlor_a").reset_index()
df = df[np.isfinite(df["chlor_a"])]
df.to_csv(os.path.join(OUT_DIR, "andaman_chl_grid_values.csv"), index=False)
print("Grid CSV saved to:", os.path.join(OUT_DIR, "andaman_chl_grid_values.csv"))

# --------- PLOT ---------
plt.figure(figsize=(8,6))
chl_clean.plot(
    robust=True,      # clip extremes for plotting
    cmap="viridis",
    cbar_kwargs={"label": "Chlorophyll-a (mg m$^{-3}$)"}
)
plt.title("MODIS Chlorophyll-a — Andaman Region")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("MODIS chlorophyll-a — Andaman region")
plt.xlabel("lon"); plt.ylabel("lat")

# Overlay reef shapefile if available
if REEF_SHAPEFILE and has_gpd:
    reefs = gpd.read_file(REEF_SHAPEFILE)
    # reproject if necessary - assuming dataset coords are in EPSG:4326
    reefs.plot(ax=plt.gca(), facecolor="none", edgecolor="red", linewidth=0.8)
    plt.legend(["Reefs"], loc="upper right")
elif REEF_SHAPEFILE and not has_gpd:
    print("Geopandas not installed; can't overlay shapefile. Install geopandas to overlay reefs.")

out_png = os.path.join(OUT_DIR, "andaman_chl_map.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("Map saved to:", out_png)
plt.show()
