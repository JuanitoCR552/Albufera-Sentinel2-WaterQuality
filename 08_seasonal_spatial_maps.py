import os
import glob
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Trebuchet MS', 'sans-serif']

# 1. File paths (replace with your local directories)
nc_folder_main = "PUT_PATH_TO_YOUR_MAIN_ACOLITE_L2W_FOLDER"
nc_folder_extra = "PUT_PATH_TO_YOUR_SECONDARY_ACOLITE_L2W_FOLDER_OR_LEAVE_SAME"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_SEASONAL_MAPS_FOLDER"

os.makedirs(output_dir, exist_ok=True)

# 2. Bio-optical parameters and custom colormaps (Figure 4A-F)
variables_info = {
    "chl_re_mishra": {
        "title": "Seasonal Spatial Distribution of Chlorophyll-a",
        "unit": "mg/m³",
        "colors": ["#FFFFFF", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
    },
    "spm_nechad2016_665": {
        "title": "Seasonal Spatial Distribution of Suspended Solids",
        "unit": "g/m³",
        "colors": ["#FFFFFF", "#f5deb3", "#d2b48c", "#8B4513", "#5c3317"]
    },
    "tur_nechad2016_665": {
        "title": "Seasonal Spatial Distribution of Turbidity",
        "unit": "NTU",
        "colors": ["#FFFFFF", "#ffeba1", "#f4a460", "#B8860B", "#8b4500"]
    },
    "p3qaa_zSD": {
        "title": "Seasonal Spatial Distribution of Secchi Disk Depth",
        "unit": "m",
        "colors": ["#FFFFFF", "#bfefff", "#5ab4ac", "#4682B4", "#003c30"]
    },
    "qaa_v6_KPAR_Lee": {
        "title": "Seasonal Spatial Distribution of Light Attenuation (K_PAR)",
        "unit": "m⁻¹",
        "colors": ["#FFFFFF", "#f3e5f5", "#ce93d8", "#9370DB", "#4a148c"]
    },
    "qaa_v6_Zeu_Lee": {
        "title": "Seasonal Spatial Distribution of Euphotic Depth (Z_eu)",
        "unit": "m",
        "colors": ["#FFFFFF", "#e0f2f1", "#80cbc4", "#20B2AA", "#004d40"]
    }
}

# 3. Meteorological seasons classification
def get_meteorological_season(month, year):
    if month in [3, 4, 5]:
        return f"Spring {year}"
    elif month in [6, 7, 8]:
        return f"Summer {year}"
    elif month in [9, 10, 11]:
        return f"Autumn {year}"
    else:
        if month == 12:
            return f"Winter {year}-{str(year + 1)[-2:]}"
        else:
            return f"Winter {year - 1}-{str(year)[-2:]}"

season_order = [
    "Spring 2023", "Summer 2023", "Autumn 2023", "Winter 2023-24",
    "Spring 2024", "Summer 2024", "Autumn 2024", "Winter 2024-25",
    "Spring 2025", "Summer 2025", "Autumn 2025", "Winter 2025-26",
    "Spring 2026"
]

def find_matching_variable(ds, target_var):
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

# 4. Clip NetCDF scenes and group spatial grids by season
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)

nc_files = sorted(list(set(
    glob.glob(os.path.join(nc_folder_main, "*L2W.nc")) +
    glob.glob(os.path.join(nc_folder_extra, "*L2W.nc"))
)))

seasonal_grids = {var: {s: [] for s in season_order} for var in variables_info.keys()}
spatial_extent = None

print(f"Processing {len(nc_files)} scenes into seasonal composites...")
for file_path in nc_files:
    base_name = os.path.basename(file_path)
    try:
        parts = base_name.split('_')
        date_str = f"{parts[2]}-{parts[3]}-{parts[4]}"
        scene_date = datetime.strptime(date_str, "%Y-%m-%d")
        season_label = get_meteorological_season(scene_date.month, scene_date.year)

        if season_label not in season_order:
            continue

        ds = xr.open_dataset(file_path)
        ds.rio.set_spatial_dims(x_dim='x' if 'x' in ds.dims else 'lon', y_dim='y' if 'y' in ds.dims else 'lat', inplace=True)
        crs_ds = ds.rio.crs or "EPSG:32630"
        if not ds.rio.crs:
            ds.rio.write_crs(crs_ds, inplace=True)
        shp_projected = lagoon_shp.to_crs(crs_ds)

        for ideal_var in variables_info.keys():
            real_var = find_matching_variable(ds, ideal_var)
            if real_var:
                clipped = ds[real_var].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True)
                seasonal_grids[ideal_var][season_label].append(clipped.values)

                if spatial_extent is None:
                    spatial_extent = [
                        clipped.x.min().item(), clipped.x.max().item(),
                        clipped.y.min().item(), clipped.y.max().item()
                    ]
        ds.close()
    except Exception:
        pass

# 5. Generate seasonal composite maps and compute lagoon-wide seasonal means
print("Generating seasonal composite maps and summary statistics...")
stats_records = []

for var, info in variables_info.items():
    clean_param_name = info["title"].replace("Seasonal Spatial Distribution of ", "")
    mean_maps = {}
    pixel_pool = []

    for season in season_order:
        grids = seasonal_grids[var][season]
        if grids:
            seasonal_mean_grid = np.nanmean(np.stack(grids, axis=0), axis=0)
            mean_maps[season] = seasonal_mean_grid
            pixel_pool.append(seasonal_mean_grid.flatten())

            stats_records.append({
                "Season": season,
                "Parameter": f"{clean_param_name} ({info['unit']})",
                "Mean_Value": np.nanmean(seasonal_mean_grid)
            })

    if not mean_maps:
        continue

    all_pixels = np.concatenate(pixel_pool)
    all_pixels = all_pixels[~np.isnan(all_pixels)]
    vmin_auto = np.maximum(0, np.percentile(all_pixels, 2))
    vmax_auto = np.percentile(all_pixels, 98)

    cmap_custom = LinearSegmentedColormap.from_list(f"Custom_{var}", info['colors'], N=256)

    valid_seasons = [s for s in season_order if s in mean_maps]
    fig, axes = plt.subplots(2, 7, figsize=(20, 5))
    fig.suptitle(info['title'], fontsize=18, fontweight='bold', y=1.02, color='#333333')
    axes = axes.flatten()
    im = None

    for i, season in enumerate(valid_seasons):
        ax = axes[i]
        im = ax.imshow(mean_maps[season], extent=spatial_extent, cmap=cmap_custom,
                       vmin=vmin_auto, vmax=vmax_auto, origin='upper')
        ax.set_title(season, fontsize=11, fontweight='bold', pad=5, color='#444444')
        ax.axis('off')

    for j in range(len(valid_seasons), len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(bottom=0.2, wspace=0.02, hspace=0.1)
    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.04])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label(f"{clean_param_name} ({info['unit']})", fontsize=12, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=10)

    output_map_path = os.path.join(output_dir, f"Seasonal_Maps_{var}.png")
    plt.savefig(output_map_path, dpi=900, bbox_inches='tight', facecolor='white')
    plt.close()

# 6. Export seasonal summary table to CSV
df_stats = pd.DataFrame(stats_records)
df_pivot = df_stats.pivot(index="Season", columns="Parameter", values="Mean_Value")
present_seasons = [s for s in season_order if s in df_pivot.index]
df_pivot = df_pivot.reindex(present_seasons)

print("\n" + df_pivot.round(3).to_string())

csv_path = os.path.join(output_dir, "Seasonal_Means_Albufera.csv")
df_pivot.to_csv(csv_path)
print(f"\nDone. Seasonal maps and CSV saved to: {output_dir}")
