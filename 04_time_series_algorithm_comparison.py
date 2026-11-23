import os
import glob
import locale
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'C')
    except Exception:
        pass

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Trebuchet MS', 'sans-serif']

# 1. File paths (replace with your local directories)
nc_folder = "PUT_PATH_TO_YOUR_ACOLITE_L2W_NETCDF_FOLDER"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_PLOTS_FOLDER"

os.makedirs(output_dir, exist_ok=True)

# 2. Bio-optical variables metadata and validation MAE values
variables_info = {
    "chl_re_mishra": {"title": "Chlorophyll-a Evolution (Mishra Algorithm)", "unit": "mg/m³", "color": "#6b8e23", "mae": 22.6, "threshold": 60},
    "chl_re_gons740": {"title": "Chlorophyll-a Evolution (Gons 740 Algorithm)", "unit": "mg/m³", "color": "#228B22", "mae": 110.0, "threshold": 60},
    "spm_nechad2016_865": {"title": "Suspended Solids Evolution (Nechad 865/866)", "unit": "g/m³", "color": "#8B4513", "mae": 29.2, "threshold": None},
    "spm_nechad2016_665": {"title": "Suspended Solids Evolution (Nechad 665/667)", "unit": "g/m³", "color": "#D2691E", "mae": 55.2, "threshold": None},
    "tur_nechad2016_665": {"title": "Turbidity Evolution (Nechad 665/667)", "unit": "NTU", "color": "#B8860B", "mae": 37.9, "threshold": None},
    "tur_dogliotti2015": {"title": "Turbidity Evolution (Dogliotti 2015)", "unit": "NTU", "color": "#A0522D", "mae": 70.0, "threshold": None},
    "tur_nechad2016_865": {"title": "Turbidity Evolution (Nechad 865/866)", "unit": "NTU", "color": "#DAA520", "mae": 30.0, "threshold": None},
    "p3qaa_zSD": {"title": "Secchi Disk Depth Evolution (QAA)", "unit": "m", "color": "#4682B4", "mae": 0.50, "threshold": None},
    "par": {"title": "Photosynthetically Active Rad. Evolution (PAR)", "unit": "µE/m²/s", "color": "#FF8C00", "mae": 0.0, "threshold": None},
    "kd_lee": {"title": "Diffuse Attenuation Evolution (Kd Lee)", "unit": "m⁻¹", "color": "#483D8B", "mae": 0.0, "threshold": None}
}

extreme_events = {
    'Valencia DANA (Oct 2024)': [pd.to_datetime('2024-10-29'), '#466380']
}

# 3. Helper function to match Sentinel-2A/2B/2C band naming variations
def find_matching_variable(ds, target_var):
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
        if 'tur' in target_lower and 'nechad' in v_l and '865' in target_lower and ('867' in v_l or '864' in v_l):
            return v
        if 'zsd' in target_lower and 'zsd' in v_l:
            return v
        if target_lower == 'par' and ('par' in v_l and 'kpar' not in v_l):
            return v
        if target_lower == 'kd_lee' and ('kpar' in v_l or 'kd' in v_l):
            return v

    return None

# 4. Clip NetCDF scenes to lagoon shapefile and compute spatial means
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)
nc_files = sorted(glob.glob(os.path.join(nc_folder, "*L2W.nc")))

print(f"Extracting lagoon-wide means from {len(nc_files)} scenes...")
time_series_records = []

for file_path in nc_files:
    try:
        base_name = os.path.basename(file_path)
        parts = base_name.split('_')
        date_str = f"{parts[2]}-{parts[3]}-{parts[4]}"
        scene_date = datetime.strptime(date_str, "%Y-%m-%d")

        ds = xr.open_dataset(file_path)
        ds.rio.set_spatial_dims(x_dim='x' if 'x' in ds.dims else 'lon', y_dim='y' if 'y' in ds.dims else 'lat', inplace=True)
        crs_ds = ds.rio.crs or "EPSG:32630"
        if not ds.rio.crs:
            ds.rio.write_crs(crs_ds, inplace=True)

        shp_projected = lagoon_shp.to_crs(crs_ds)
        daily_data = {'Date': scene_date}

        for ideal_var in variables_info.keys():
            real_var = find_matching_variable(ds, ideal_var)
            if real_var:
                clipped = ds[real_var].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True)
                daily_data[ideal_var] = np.nanmean(clipped.values)
            else:
                daily_data[ideal_var] = np.nan

        time_series_records.append(daily_data)
        ds.close()
    except Exception:
        pass

df = pd.DataFrame(time_series_records).sort_values('Date').reset_index(drop=True)

# 5. Plot multi-algorithm time-series comparisons
def plot_algorithm_comparison(vars_list, title, ylabel, output_filename):
    existing_vars = [v for v in vars_list if v in df.columns]
    if len(existing_vars) < 2:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=35, loc='left')
    legend_handles = []

    for var in existing_vars:
        df_clean = df[['Date', var]].dropna()
        if not df_clean.empty:
            ax.plot(df_clean['Date'], df_clean[var], marker='o', markersize=4,
                    linestyle='-', linewidth=1.5, color=variables_info[var]['color'])
            label_str = variables_info[var]['title'].split(' (')[1].replace(')', '')
            legend_handles.append(Line2D([0], [0], color=variables_info[var]['color'], marker='o', label=label_str))

    for event_name, event_meta in extreme_events.items():
        if df['Date'].min() <= event_meta[0] <= df['Date'].max():
            ax.axvline(x=event_meta[0], color=event_meta[1], linestyle='--', alpha=0.9)
            legend_handles.append(Line2D([0], [0], color=event_meta[1], linestyle='--', label=event_name))

    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', which='minor', labelsize=10, labelcolor='#777777', rotation=45)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', which='major', labelsize=12, labelcolor='black', pad=25)

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='#333333')

    unique_handles = list({h.get_label(): h for h in legend_handles}.values())
    ax.legend(handles=unique_handles, loc='lower right', bbox_to_anchor=(1.0, 1.02),
              ncol=len(unique_handles), framealpha=0.9, edgecolor='#cccccc', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Comparison_{output_filename}.png"), dpi=500, bbox_inches='tight')
    plt.close()

print("Generating multi-year algorithm comparison plots...")
plot_algorithm_comparison(["spm_nechad2016_865", "spm_nechad2016_665"],
                          "Algorithm Comparison: Suspended Solids (Nechad)", "SPM (g/m³)", "SPM_Algorithms")
plot_algorithm_comparison(["tur_nechad2016_665", "tur_nechad2016_865"],
                          "Algorithm Comparison: Turbidity (Nechad)", "Turbidity (NTU)", "TUR_Algorithms_Nechad")

print(f"Done. Plots saved to: {output_dir}")
