import os
import re
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

try:
    locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'C')
    except Exception:
        pass

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Trebuchet MS', 'Arial', 'sans-serif']

# 1. File paths (replace with your local directories)
nc_folder_main = "PUT_PATH_TO_YOUR_MAIN_ACOLITE_L2W_FOLDER"
nc_folder_extra = "PUT_PATH_TO_YOUR_SECONDARY_ACOLITE_L2W_FOLDER_OR_LEAVE_SAME"
shapefile_path = "PUT_PATH_TO_YOUR_ALBUFERA_SHAPEFILE.shp"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_FOLDER"

os.makedirs(output_dir, exist_ok=True)

# 2. Bio-optical and optical parameters configuration (Figure 3A-F + NIR comparisons)
variables_config = {
    "chl_re_mishra": {
        "title": "Chlorophyll-a Evolution (Mishra Algorithm)",
        "ylabel": "Chlorophyll-a (mg/m³)",
        "color": "#6b8e23",
        "band_type": "mae",
        "mae": 22.6,
        "hypertrophic_threshold": 60.0,
        "plot_individual": True
    },
    "spm_nechad2016_665": {
        "title": "Suspended Particulate Matter (SPM) - Nechad2016_665",
        "ylabel": "Suspended Particulate Matter (SPM) (mg/L)",
        "color": "#D2691E",
        "band_type": "mae",
        "mae": 55.2,
        "hypertrophic_threshold": None,
        "plot_individual": True
    },
    "tur_nechad2016_665": {
        "title": "Turbidity (TUR) - Nechad2016_667",
        "ylabel": "Turbidity (TUR) (NTU)",
        "color": "#B8860B",
        "band_type": "mae",
        "mae": 37.9,
        "hypertrophic_threshold": None,
        "plot_individual": True
    },
    "p3qaa_zSD": {
        "title": "Secchi Disk Depth (ZSD) - p3qaa",
        "ylabel": "Secchi Disk Depth (ZSD) (m)",
        "color": "#2980b9",
        "band_type": "mae",
        "mae": 0.50,
        "hypertrophic_threshold": None,
        "plot_individual": True
    },
    "qaa_v6_Zeu_Lee": {
        "title": "Euphotic Depth (Zeu) - Lee",
        "ylabel": "Euphotic Depth (Zeu) (m)",
        "color": "#20B2AA",
        "band_type": "std",
        "mae": None,
        "hypertrophic_threshold": None,
        "plot_individual": True
    },
    "qaa_v6_KdPAR_Nechad": {
        "title": "Diffuse Attenuation Coefficient (K_PAR)",
        "ylabel": "Diffuse Attenuation Coefficient (K_PAR) (m⁻¹)",
        "color": "#9370DB",
        "band_type": "std",
        "mae": None,
        "hypertrophic_threshold": None,
        "plot_individual": True
    },
    # Additional NIR algorithms used for multi-year stability comparison plots
    "spm_nechad2016_865": {
        "title": "Suspended Particulate Matter (SPM) - Nechad2016_865",
        "ylabel": "SPM (mg/L)",
        "color": "#8B4513",
        "band_type": "mae",
        "mae": 29.2,
        "hypertrophic_threshold": None,
        "plot_individual": False
    },
    "tur_nechad2016_865": {
        "title": "Turbidity (TUR) - Nechad2016_865",
        "ylabel": "Turbidity (NTU)",
        "color": "#DAA520",
        "band_type": "mae",
        "mae": 30.0,
        "hypertrophic_threshold": None,
        "plot_individual": False
    }
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
        if 'spm' in target_lower and '865' in target_lower and 'spm' in v_l and any(b in v_l for b in ['864', '865', '866', '867']):
            return v
        if 'spm' in target_lower and '665' in target_lower and 'spm' in v_l and ('665' in v_l or '667' in v_l):
            return v
        if 'tur' in target_lower and '865' in target_lower and 'tur' in v_l and any(b in v_l for b in ['864', '865', '866', '867']):
            return v
        if 'tur' in target_lower and '665' in target_lower and 'tur' in v_l and ('665' in v_l or '667' in v_l):
            return v
        if 'zsd' in target_lower and 'zsd' in v_l:
            return v
        if 'zeu' in target_lower and 'zeu' in v_l:
            return v
        if 'kdpar' in target_lower and ('kpar' in v_l or 'kd' in v_l):
            return v

    return None

# 4. Clip NetCDF scenes to lagoon shapefile and extract full time series
print("Loading lagoon shapefile...")
lagoon_shp = gpd.read_file(shapefile_path)

nc_files = sorted(list(set(
    glob.glob(os.path.join(nc_folder_main, "*L2W.nc")) +
    glob.glob(os.path.join(nc_folder_extra, "*L2W.nc"))
)))

if not nc_files:
    print("No *L2W.nc files found in the specified directories.")
    exit()

print(f"Extracting multi-year time series from {len(nc_files)} scenes...")
merged_records = {}

for file_path in nc_files:
    try:
        base_name = os.path.basename(file_path)
        match = re.search(r"(\d{4}_\d{2}_\d{2})", base_name)
        if not match:
            continue
        date_str = match.group(1).replace('_', '-')

        if date_str not in merged_records:
            merged_records[date_str] = {'Date': datetime.strptime(date_str, "%Y-%m-%d")}

        ds = xr.open_dataset(file_path)
        ds.rio.set_spatial_dims(x_dim='x' if 'x' in ds.dims else 'lon', y_dim='y' if 'y' in ds.dims else 'lat', inplace=True)
        crs_ds = ds.rio.crs or "EPSG:32630"
        if not ds.rio.crs:
            ds.rio.write_crs(crs_ds, inplace=True)

        shp_projected = lagoon_shp.to_crs(crs_ds)

        for ideal_var in variables_config.keys():
            real_var = find_matching_variable(ds, ideal_var)
            if real_var:
                clipped = ds[real_var].rio.clip(shp_projected.geometry, shp_projected.crs, drop=True)
                mean_val = np.nanmean(clipped.values)
                std_val = np.nanstd(clipped.values)

                # Filter extreme atmospheric/sunglint outliers in K_PAR
                if 'kdpar' in ideal_var.lower() and mean_val > 40:
                    mean_val = np.nan
                    std_val = np.nan

                merged_records[date_str][ideal_var] = mean_val
                merged_records[date_str][f"{ideal_var}_std"] = std_val

        ds.close()
    except Exception:
        pass

df = pd.DataFrame(list(merged_records.values())).sort_values('Date').reset_index(drop=True)

csv_output = os.path.join(output_dir, "Albufera_WaterQuality_TimeSeries_2023_2026.csv")
df.to_csv(csv_output, index=False)
print(f"Time-series dataset exported to: {csv_output}")

# 5. Plot individual time series for all parameters (Figure 3A-F)
def plot_individual_time_series(df_data, var):
    info = variables_config[var]
    if not info.get("plot_individual", True):
        return
    if var not in df_data.columns or df_data[var].dropna().empty:
        print(f"No valid data for {var}, skipping plot.")
        return

    std_col = f"{var}_std"
    cols_needed = ['Date', var, std_col] if std_col in df_data.columns else ['Date', var]
    df_var = df_data[cols_needed].dropna(subset=['Date', var])

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(info['title'], fontsize=16, fontweight='bold', pad=35, loc='left', color='#333333')

    # Uncertainty / spatial variability shaded band
    if info['band_type'] == 'mae' and info['mae'] is not None:
        y_lower = np.maximum(0, df_var[var] - info['mae'])
        y_upper = df_var[var] + info['mae']
        ax.fill_between(df_var['Date'], y_lower, y_upper, color=info['color'], alpha=0.15, zorder=2, edgecolor='none')
        band_label = f"MAE (±{info['mae']})"
    elif info['band_type'] == 'std' and std_col in df_var.columns:
        y_lower = np.maximum(0, df_var[var] - df_var[std_col])
        y_upper = df_var[var] + df_var[std_col]
        ax.fill_between(df_var['Date'], y_lower, y_upper, color=info['color'], alpha=0.15, zorder=2, edgecolor='none')
        band_label = "Spatial Std Dev"
    else:
        y_upper = df_var[var]
        band_label = None

    # Main time-series line
    ax.plot(df_var['Date'], df_var[var], marker='o', markersize=4,
            linestyle='-', linewidth=1.5, color=info['color'], zorder=3)

    max_y = y_upper.max() * 1.1
    ax.set_ylim(0, max_y)

    # Trophic state background shading (Chlorophyll-a > 60 mg/m³)
    threshold = info.get('hypertrophic_threshold')
    if threshold is not None:
        ax.axhspan(threshold, max_y, color='#e5e5e5', alpha=0.35, zorder=0)
        ax.axhline(threshold, color='#b0b0b0', linestyle=':', linewidth=1.0, zorder=1)

    # Extreme event vertical marker (Valencia DANA)
    for event_meta in extreme_events.values():
        if df_var['Date'].min() <= event_meta[0] <= df_var['Date'].max():
            ax.axvline(x=event_meta[0], color=event_meta[1], linestyle='--', linewidth=1.5, alpha=0.9, zorder=1)

    # Annotate prominent peaks
    try:
        peak_indices, _ = find_peaks(df_var[var], distance=4, prominence=df_var[var].max() * 0.15)
        for _, row in df_var.iloc[peak_indices].iterrows():
            ax.annotate(f"{row[var]:.1f}", xy=(row['Date'], row[var]), xytext=(0, 6), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=info['color'])
    except Exception:
        pass

    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', which='minor', labelsize=10, labelcolor='#777777', rotation=45)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', which='major', labelsize=12, labelcolor='black', pad=25)
    for label in ax.get_xticklabels(which='major'):
        label.set_fontweight('bold')

    ax.set_ylabel(info['ylabel'], fontsize=12, fontweight='bold', color='#333333')

    legend_handles = [
        Line2D([0], [0], color=info['color'], marker='o', markersize=5, linestyle='-', label='Albufera Mean')
    ]
    if band_label:
        legend_handles.append(Patch(facecolor=info['color'], alpha=0.15, label=band_label, edgecolor='none'))
    if threshold is not None:
        legend_handles.append(Patch(facecolor='#e5e5e5', alpha=0.5, label='Hypertrophic State (>60 mg/m³)', edgecolor='#cccccc'))
        legend_handles.append(Patch(facecolor='#ffffff', label='Eutrophic State (<60 mg/m³)', edgecolor='#cccccc'))
    for event_name, event_meta in extreme_events.items():
        legend_handles.append(Line2D([0], [0], color=event_meta[1], linestyle='--', linewidth=1.5, label=event_name))

    ax.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(1.0, 1.02),
              ncol=3 if threshold else len(legend_handles), framealpha=0.9, edgecolor='#cccccc', fontsize=9.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"TimeSeries_{var}.png"), dpi=600, bbox_inches='tight')
    plt.close()

# 6. Plot NIR vs. Red algorithm stability comparisons
def plot_algorithm_comparison(df_data, vars_list, title, ylabel, filename):
    existing_vars = [v for v in vars_list if v in df_data.columns]
    if len(existing_vars) < 2:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=35, loc='left', color='#333333')
    legend_handles = []

    for var in existing_vars:
        df_clean = df_data[['Date', var]].dropna()
        if not df_clean.empty:
            ax.plot(df_clean['Date'], df_clean[var], marker='o', markersize=4,
                    linestyle='-', linewidth=1.5, color=variables_config[var]['color'])
            label_str = variables_config[var]['title'].split(' - ')[-1]
            legend_handles.append(Line2D([0], [0], color=variables_config[var]['color'], marker='o', label=label_str))

    for event_name, event_meta in extreme_events.items():
        if df_data['Date'].min() <= event_meta[0] <= df_data['Date'].max():
            ax.axvline(x=event_meta[0], color=event_meta[1], linestyle='--', linewidth=1.5, alpha=0.9)
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
    plt.savefig(os.path.join(output_dir, f"Comparison_{filename}.png"), dpi=500, bbox_inches='tight')
    plt.close()

print("Generating time-series and algorithm comparison plots...")
for var in variables_config.keys():
    plot_individual_time_series(df, var)

plot_algorithm_comparison(df, ["spm_nechad2016_865", "spm_nechad2016_665"],
                          "Algorithm Comparison: Suspended Particulate Matter (Nechad)", "SPM (mg/L)", "SPM_Algorithms")
plot_algorithm_comparison(df, ["tur_nechad2016_665", "tur_nechad2016_865"],
                          "Algorithm Comparison: Turbidity (Nechad)", "Turbidity (NTU)", "TUR_Algorithms_Nechad")

print(f"Done. All outputs saved to: {output_dir}")
