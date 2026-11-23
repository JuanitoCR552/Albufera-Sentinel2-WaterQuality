import os
import glob
import warnings
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Trebuchet MS', 'sans-serif']

# 1. File paths and target dates (replace with your local directories)
nc_folder_main = "PUT_PATH_TO_YOUR_MAIN_ACOLITE_L2W_FOLDER"
nc_folder_extra = "PUT_PATH_TO_YOUR_SECONDARY_ACOLITE_L2W_FOLDER_OR_LEAVE_SAME"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_MAPS_FOLDER"

os.makedirs(output_dir, exist_ok=True)
POST_DANA_DATE = "2024-11-10"

# 2. Bio-optical parameters and custom colormaps
parameters = {
    "CHL": {
        "sat_var": "chl_re_mishra", "title": "Chlorophyll-a", "unit": "mg/m³",
        "abs_colors": ["#FFFFFF", "#c7e9c0", "#74c476", "#238b45", "#00441b"], "anomaly_cmap": "RdBu_r"
    },
    "SPM": {
        "sat_var": "spm_nechad2016_665", "title": "Suspended Solids (SPM)", "unit": "g/m³",
        "abs_colors": ["#FFFFFF", "#f5deb3", "#d2b48c", "#8B4513", "#5c3317"], "anomaly_cmap": "RdBu_r"
    },
    "TUR": {
        "sat_var": "tur_nechad2016_665", "title": "Turbidity", "unit": "NTU",
        "abs_colors": ["#FFFFFF", "#ffeba1", "#f4a460", "#B8860B", "#8b4500"], "anomaly_cmap": "RdBu_r"
    },
    "KPAR": {
        "sat_var": "qaa_v6_KPAR_Lee", "title": "Light Attenuation (K_PAR)", "unit": "m⁻¹",
        "abs_colors": ["#FFFFFF", "#f3e5f5", "#ce93d8", "#9370DB", "#4a148c"], "anomaly_cmap": "RdBu_r"
    },
    "ZSD": {
        "sat_var": "p3qaa_zSD", "title": "Secchi Disk Depth", "unit": "m",
        "abs_colors": ["#FFFFFF", "#bfefff", "#5ab4ac", "#4682B4", "#003c30"], "anomaly_cmap": "RdBu"
    },
    "ZEU": {
        "sat_var": "qaa_v6_Zeu_Lee", "title": "Euphotic Depth (Z_eu)", "unit": "m",
        "abs_colors": ["#FFFFFF", "#e0f2f1", "#80cbc4", "#20B2AA", "#004d40"], "anomaly_cmap": "RdBu"
    }
}

# 3. Helper function to handle Sentinel-2A/2B/2C band naming variations
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
        if 'zsd' in target_lower and 'zsd' in v_l:
            return v
    return None

# 4. Clip NetCDF scenes to lagoon shapefile for baseline and post-DANA dates
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)

nc_files = sorted(list(set(
    glob.glob(os.path.join(nc_folder_main, "*L2W.nc")) +
    glob.glob(os.path.join(nc_folder_extra, "*L2W.nc"))
)))

sept_arrays = {key: [] for key in parameters.keys()}
post_arrays = {key: None for key in parameters.keys()}
spatial_extent = None

print(f"Extracting spatial grids (September 2024 baseline vs. {POST_DANA_DATE})...")

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
                current_grid = clipped.values

                if spatial_extent is None:
                    spatial_extent = [
                        clipped.x.min().item(), clipped.x.max().item(),
                        clipped.y.min().item(), clipped.y.max().item()
                    ]

                if year == "2024" and month == "09":
                    sept_arrays[key].append(current_grid)
                elif date_str == POST_DANA_DATE:
                    post_arrays[key] = current_grid

        ds.close()
        print(f"Loaded scene: {date_str}")
    except Exception:
        pass

# 5. Generate 3-panel DANA impact anomaly maps (Baseline, Post-Event, Anomaly)
print("Generating 3-panel DANA impact maps...")

for key, info in parameters.items():
    pre_grids = sept_arrays[key]
    post_grid = post_arrays[key]

    if not pre_grids or post_grid is None:
        print(f"Warning: Missing data for {info['title']}, skipping map.")
        continue

    pre_mean_grid = np.nanmean(np.stack(pre_grids, axis=0), axis=0)
    anomaly_grid = post_grid - pre_mean_grid

    cmap_custom_abs = LinearSegmentedColormap.from_list(f"Custom_{key}", info['abs_colors'], N=256)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"DANA Impact on {info['title']}", fontsize=20, fontweight='bold', y=0.98, color='#333333')

    all_vals = np.concatenate([pre_mean_grid.flatten(), post_grid.flatten()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin_abs, vmax_abs = np.maximum(0, np.percentile(all_vals, 2)), np.percentile(all_vals, 98)

    # Panel 1: Pre-DANA Baseline
    axes[0].imshow(pre_mean_grid, extent=spatial_extent, cmap=cmap_custom_abs, vmin=vmin_abs, vmax=vmax_abs, origin='upper')
    axes[0].set_title("Baseline (September 2024 Avg)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Post-DANA
    im2 = axes[1].imshow(post_grid, extent=spatial_extent, cmap=cmap_custom_abs, vmin=vmin_abs, vmax=vmax_abs, origin='upper')
    axes[1].set_title(f"Post-Event ({POST_DANA_DATE})", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Impact Anomaly (Post - Baseline)
    anom_limit = np.nanpercentile(np.abs(anomaly_grid), 98)
    im3 = axes[2].imshow(anomaly_grid, extent=spatial_extent, cmap=info['anomaly_cmap'], vmin=-anom_limit, vmax=anom_limit, origin='upper')
    axes[2].set_title("Impact Anomaly (Post - Baseline)", fontsize=14, fontweight='bold', color='#8b0000')
    axes[2].axis('off')

    cbar_ax1 = fig.add_axes([0.15, 0.1, 0.4, 0.03])
    cbar1 = fig.colorbar(im2, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(f"Concentration / Depth ({info['unit']})", fontsize=12, fontweight='bold')

    cbar_ax2 = fig.add_axes([0.65, 0.1, 0.25, 0.03])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='horizontal', extend='both')
    cbar2.set_label(f"Change ({info['unit']})", fontsize=12, fontweight='bold')

    plt.subplots_adjust(bottom=0.2)
    output_path = os.path.join(output_dir, f"DANA_Impact_Map_{key}.png")
    plt.savefig(output_path, dpi=900, bbox_inches='tight', facecolor='white')
    plt.close()

print(f"Done. Maps saved to: {output_dir}")
