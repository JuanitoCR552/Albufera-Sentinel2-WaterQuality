import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr

warnings.filterwarnings("ignore")

# 1. File paths and target dates (replace with your local directories)
nc_folder_main = "PUT_PATH_TO_YOUR_MAIN_ACOLITE_L2W_FOLDER"
nc_folder_extra = "PUT_PATH_TO_YOUR_SECONDARY_ACOLITE_L2W_FOLDER_OR_LEAVE_SAME"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_FOLDER"

os.makedirs(output_dir, exist_ok=True)
POST_DANA_DATE = "2024-11-10"

# 2. Bio-optical parameters configuration
parameters = {
    "CHL": {"sat_var": "chl_re_mishra", "title": "Chlorophyll-a", "unit": "mg/m³"},
    "SPM": {"sat_var": "spm_nechad2016_665", "title": "Suspended Solids (SPM)", "unit": "g/m³"},
    "TUR": {"sat_var": "tur_nechad2016_665", "title": "Turbidity", "unit": "NTU"},
    "KPAR": {"sat_var": "qaa_v6_KPAR_Lee", "title": "Light Attenuation (K_PAR)", "unit": "m⁻¹"},
    "ZSD": {"sat_var": "p3qaa_zSD", "title": "Secchi Disk Depth", "unit": "m"},
    "ZEU": {"sat_var": "qaa_v6_Zeu_Lee", "title": "Euphotic Depth (Z_eu)", "unit": "m"}
}

# 3. Helper function to handle Sentinel-2A/2B/2C band naming variations
def find_var(ds, target_var):
    target_lower = target_var.lower()

    for v in ds.variables:
        if target_lower == v.lower():
            return v

    for v in ds.variables:
        v_l = v.lower()
        if '865' in target_lower and ('867' in v_l or '864' in v_l or '866' in v_l) and 'spm' in v_l:
            return v
        if '867' in target_lower and ('865' in v_l or '864' in v_l or '866' in v_l) and 'spm' in v_l:
            return v
        if '665' in target_lower and '667' in v_l:
            return v
        if '667' in target_lower and '665' in v_l:
            return v
        if 'tur' in target_lower and 'dogliotti' in v_l:
            return v
        if 'tur' in target_lower and 'nechad' in v_l:
            return v
        if 'zsd' in target_lower and 'zsd' in v_l:
            return v

    return None

# 4. Clip scenes and extract pre-DANA (September 2024) and post-DANA arrays
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)

nc_files = sorted(list(set(
    glob.glob(os.path.join(nc_folder_main, "*L2W.nc")) +
    glob.glob(os.path.join(nc_folder_extra, "*L2W.nc"))
)))

sept_arrays = {key: [] for key in parameters.keys()}
post_arrays = {key: None for key in parameters.keys()}

print(f"Extracting DANA impact metrics (September 2024 baseline vs. {POST_DANA_DATE})...")

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

        for key, info in parameters.items():
            real_var = find_var(ds, info["sat_var"])
            if real_var:
                clipped = ds[real_var].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True)
                if year == "2024" and month == "09":
                    sept_arrays[key].append(clipped.values)
                elif date_str == POST_DANA_DATE:
                    post_arrays[key] = clipped.values
        ds.close()
    except Exception:
        pass

# 5. Compute lagoon-wide means, absolute change, and percentage variation
results = []

for key, info in parameters.items():
    if not sept_arrays[key] or post_arrays[key] is None:
        print(f"Warning: Missing data for {info['title']}.")
        continue

    pre_mean_grid = np.nanmean(np.stack(sept_arrays[key], axis=0), axis=0)
    post_grid = post_arrays[key]

    mean_pre = np.nanmean(pre_mean_grid)
    mean_post = np.nanmean(post_grid)

    abs_change = mean_post - mean_pre
    pct_change = (abs_change / mean_pre) * 100.0

    results.append({
        "Parameter": info["title"],
        "Baseline (Sept 2024)": round(mean_pre, 3),
        "Post-DANA (Nov 2024)": round(mean_post, 3),
        "Unit": info["unit"],
        "Absolute Change": round(abs_change, 3),
        "Change (%)": round(pct_change, 1)
    })

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

csv_path = os.path.join(output_dir, "DANA_Impact_Statistics.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")
