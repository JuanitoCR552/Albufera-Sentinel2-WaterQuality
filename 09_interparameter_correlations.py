import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
from scipy.stats import linregress

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Trebuchet MS', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.5

# 1. File paths (replace with your local directories)
nc_folder_main = "PUT_PATH_TO_YOUR_MAIN_ACOLITE_L2W_FOLDER"
nc_folder_extra = "PUT_PATH_TO_YOUR_SECONDARY_ACOLITE_L2W_FOLDER_OR_LEAVE_SAME"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_CORRELATIONS_FOLDER"

os.makedirs(output_dir, exist_ok=True)

# 2. Helper function to match Sentinel-2A/2B/2C band naming variations
def find_var(ds, target_var):
    target_lower = target_var.lower()
    for v in ds.variables:
        if target_lower == v.lower():
            return v

    for v in ds.variables:
        v_l = v.lower()
        if '665' in target_lower and '667' in v_l:
            return v
        if '667' in target_lower and '665' in v_l:
            return v
        if '865' in target_lower and ('867' in v_l or '864' in v_l or '866' in v_l) and 'spm' in v_l:
            return v
        if 'tur' in target_lower and 'nechad' in v_l and ('665' in v_l or '667' in v_l):
            return v
        if 'tur' in target_lower and 'dogliotti' in v_l:
            return v
        if 'zsd' in target_lower and 'zsd' in v_l:
            return v
    return None

# 3. Extract whole-lagoon means for all valid satellite dates
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)

nc_files = sorted(list(set(
    glob.glob(os.path.join(nc_folder_main, "*L2W.nc")) +
    glob.glob(os.path.join(nc_folder_extra, "*L2W.nc"))
)))

merged_records = {}

print(f"Extracting whole-lagoon means from {len(nc_files)} scenes...")
for file_path in nc_files:
    try:
        base_name = os.path.basename(file_path)
        match = re.search(r"(\d{4}_\d{2}_\d{2})", base_name)
        if not match:
            continue
        date_str = match.group(1).replace('_', '-')

        if date_str not in merged_records:
            merged_records[date_str] = {'Date': date_str}

        ds = xr.open_dataset(file_path)
        ds.rio.set_spatial_dims(x_dim='x' if 'x' in ds.dims else 'lon', y_dim='y' if 'y' in ds.dims else 'lat', inplace=True)
        crs_ds = ds.rio.crs or "EPSG:32630"
        if not ds.rio.crs:
            ds.rio.write_crs(crs_ds, inplace=True)
        shp_projected = lagoon_shp.to_crs(crs_ds)

        v_chl = find_var(ds, "chl_re_mishra")
        if v_chl:
            merged_records[date_str]['CHL'] = np.nanmean(ds[v_chl].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True).values)

        v_spm = find_var(ds, "spm_nechad2016_665") or find_var(ds, "spm_nechad2016_865")
        if v_spm:
            merged_records[date_str]['SPM'] = np.nanmean(ds[v_spm].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True).values)

        v_tur = find_var(ds, "tur_nechad2016_665") or find_var(ds, "tur_dogliotti2015")
        if v_tur:
            merged_records[date_str]['TUR'] = np.nanmean(ds[v_tur].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True).values)

        v_zsd = find_var(ds, "p3qaa_zSD") or find_var(ds, "zsd_lee")
        if v_zsd:
            merged_records[date_str]['ZSD'] = np.nanmean(ds[v_zsd].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True).values)

        v_kpar = find_var(ds, "qaa_v6_KPAR_Lee")
        if v_kpar:
            merged_records[date_str]['KPAR'] = np.nanmean(ds[v_kpar].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True).values)

        v_zeu = find_var(ds, "qaa_v6_Zeu_Lee")
        if v_zeu:
            merged_records[date_str]['ZEU'] = np.nanmean(ds[v_zeu].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True).values)

        ds.close()
    except Exception:
        pass

df = pd.DataFrame(list(merged_records.values()))

# 4. Pairwise scatter plots and linear regressions (Figure 6A-F)
def plot_pairwise_correlation(x_col, y_col, x_label, y_label, title, point_color, filename):
    if x_col not in df.columns or y_col not in df.columns:
        return
    df_pair = df[[x_col, y_col]].dropna()

    # Remove extreme outliers using 3x IQR filter
    for col in [x_col, y_col]:
        q1 = df_pair[col].quantile(0.25)
        q3 = df_pair[col].quantile(0.75)
        iqr = q3 - q1
        df_pair = df_pair[(df_pair[col] >= q1 - 3 * iqr) & (df_pair[col] <= q3 + 3 * iqr)]

    if len(df_pair) < 3:
        print(f"Insufficient valid pairs for {x_col} vs. {y_col}.")
        return

    x = df_pair[x_col]
    y = df_pair[y_col]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(True, linestyle='--', alpha=0.4, color='#b0bec5', zorder=0)
    ax.scatter(x, y, color=point_color, alpha=0.85, edgecolors='white', linewidth=1.2, s=100, zorder=3)

    slope, intercept, r_value, _, _ = linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept

    reg_label = f"R² = {r_value**2:.3f}\ny = {slope:.2f}x + {intercept:.2f}"
    ax.plot(line_x, line_y, color='#2c3e50', linestyle='--', linewidth=2.5, zorder=4, label=reg_label)

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#222222')
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold', labelpad=12, color='#333333')
    ax.set_ylabel(y_label, fontsize=15, fontweight='bold', labelpad=12, color='#333333')

    legend = ax.legend(loc='best', fontsize=13, framealpha=0.95, edgecolor='#dddddd')
    legend.get_frame().set_linewidth(1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333', length=6, width=1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=800, bbox_inches='tight', facecolor='white')
    plt.close()

print("Generating inter-parameter correlation scatter plots...")
plot_pairwise_correlation('SPM', 'TUR', "Suspended Solids (g/m³)", "Turbidity (NTU)", "Correlation: Turbidity vs. SPM", "#8B4513", "1_Corr_TUR_vs_SPM")
plot_pairwise_correlation('CHL', 'TUR', "Chlorophyll-a (mg/m³)", "Turbidity (NTU)", "Correlation: Turbidity vs. Chlorophyll-a", "#228B22", "2_Corr_TUR_vs_CHL")
plot_pairwise_correlation('SPM', 'ZEU', "Suspended Solids (g/m³)", "Euphotic Depth Z_eu (m)", "Correlation: Euphotic Depth vs. SPM", "#20B2AA", "3_Corr_ZEU_vs_SPM")
plot_pairwise_correlation('CHL', 'ZEU', "Chlorophyll-a (mg/m³)", "Euphotic Depth Z_eu (m)", "Correlation: Euphotic Depth vs. Chlorophyll-a", "#9370DB", "4_Corr_ZEU_vs_CHL")
plot_pairwise_correlation('CHL', 'KPAR', "Chlorophyll-a (mg/m³)", "Light Attenuation K_PAR (m⁻¹)", "Correlation: K_PAR vs. Chlorophyll-a", "#B22222", "5_Corr_KPAR_vs_CHL")
plot_pairwise_correlation('SPM', 'ZSD', "Suspended Solids (g/m³)", "Secchi Disk Depth (m)", "Correlation: Secchi Depth vs. SPM", "#4682B4", "6_Corr_ZSD_vs_SPM")

df.to_csv(os.path.join(output_dir, "Interparameter_Correlations_Dataset.csv"), index=False)
print(f"Done. Scatter plots and dataset saved to: {output_dir}")
