import os
import glob
import warnings
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rxr

warnings.filterwarnings("ignore")

# 1. File paths and target dates (replace with your local directories)
nc_folder_main = "PUT_PATH_TO_YOUR_MAIN_ACOLITE_L2W_FOLDER"
nc_folder_extra = "PUT_PATH_TO_YOUR_SECONDARY_ACOLITE_L2W_FOLDER_OR_LEAVE_SAME"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"

POST_DANA_DATE = "2024-11-10"

# 2. Helper function to match Sentinel-2A/2B/2C band naming variations
def find_var(ds, target_var):
    target_lower = target_var.lower()
    for v in ds.variables:
        v_l = v.lower()
        if target_lower == v_l:
            return v
        if '665' in target_lower and target_lower.replace('665', '667') == v_l:
            return v
        if '667' in target_lower and target_lower.replace('667', '665') == v_l:
            return v
    return None

# 3. Extract spatial arrays for September 2024 baseline and post-DANA scene
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)

nc_files = sorted(list(set(
    glob.glob(os.path.join(nc_folder_main, "*L2W.nc")) +
    glob.glob(os.path.join(nc_folder_extra, "*L2W.nc"))
)))

september_spm_arrays = []
post_dana_spm_array = None

print("Processing SPM scenes (September 2024 baseline vs. Post-DANA)...")

for file_path in nc_files:
    base_name = os.path.basename(file_path)
    try:
        parts = base_name.split('_')
        year, month, day = parts[2], parts[3], parts[4]
        date_str = f"{year}-{month}-{day}"

        if not ((year == "2024" and month == "09") or date_str == POST_DANA_DATE):
            continue

        ds = xr.open_dataset(file_path)
        ds.rio.set_spatial_dims(x_dim='x' if 'x' in ds.dims else 'lon', y_dim='y' if 'y' in ds.dims else 'lat', inplace=True)
        crs_ds = ds.rio.crs or "EPSG:32630"
        if not ds.rio.crs:
            ds.rio.write_crs(crs_ds, inplace=True)
        shp_projected = lagoon_shp.to_crs(crs_ds)

        var_spm = find_var(ds, "spm_nechad2016_665")

        if var_spm:
            clipped = ds[var_spm].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True)
            current_array = clipped.values

            if year == "2024" and month == "09":
                september_spm_arrays.append(current_array)
            elif date_str == POST_DANA_DATE:
                post_dana_spm_array = current_array

        ds.close()
    except Exception:
        pass

# 4. Compute lagoon-wide spatial mean and standard deviation
print("\n--- DANA Impact SPM Statistics ---")

if september_spm_arrays:
    sept_temporal_mean = np.nanmean(np.stack(september_spm_arrays, axis=0), axis=0)
    sept_valid_pixels = sept_temporal_mean[~np.isnan(sept_temporal_mean)]

    mean_sept = np.nanmean(sept_valid_pixels)
    std_sept = np.nanstd(sept_valid_pixels)

    print(f"Baseline (September 2024): {mean_sept:.2f} ± {std_sept:.2f} mg/L")
else:
    print("No valid scenes found for September 2024.")

if post_dana_spm_array is not None:
    post_valid_pixels = post_dana_spm_array[~np.isnan(post_dana_spm_array)]
    mean_post = np.nanmean(post_valid_pixels)
    std_post = np.nanstd(post_valid_pixels)

    print(f"Post-DANA ({POST_DANA_DATE}): {mean_post:.2f} ± {std_post:.2f} mg/L")
else:
    print(f"No valid scene found for {POST_DANA_DATE}.")

if september_spm_arrays and post_dana_spm_array is not None:
    net_increase = mean_post - mean_sept
    pct_increase = (net_increase / mean_sept) * 100.0
    print(f"Net SPM Increase: +{net_increase:.2f} mg/L (+{pct_increase:.1f}%)")
