import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. File paths (replace with your local directories)
excel_path = "PUT_PATH_TO_YOUR_IN_SITU_EXCEL_FILE.xlsx"
nc_path = "PUT_PATH_TO_YOUR_ACOLITE_L2W_NETCDF_FILE.nc"
output_dir = "PUT_PATH_TO_YOUR_OUTPUT_FOLDER"

os.makedirs(output_dir, exist_ok=True)
base_name = os.path.basename(nc_path).replace(".nc", "")

# 2. Load in situ Excel and ACOLITE NetCDF data
# Expected Excel columns: Punto_ID, Lat, Lon, Clorofila_LAB, SS_LAB, TUR_LAB, Secchi_LAB
print("Loading in situ data and Sentinel-2 L2W NetCDF...")
df_lab = pd.read_excel(excel_path, decimal=',')
df_lab.columns = df_lab.columns.str.strip()

# Clean coordinates (handle comma decimals if present)
df_lab['Lat'] = pd.to_numeric(df_lab['Lat'].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
df_lab['Lon'] = pd.to_numeric(df_lab['Lon'].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
df_lab = df_lab.dropna(subset=['Lat', 'Lon'])

if 'Secchi_LAB' not in df_lab.columns:
    print("Warning: 'Secchi_LAB' column not found in the Excel file.")

ds = xr.open_dataset(nc_path)
lat_sat = ds['lat'].values
lon_sat = ds['lon'].values

keywords = ['chl', 'spm', 'tur', 'zsd', 'kpar', 'zeu', 'fai', 'ndvi']
acolite_vars = [v for v in ds.variables if any(k in v.lower() for k in keywords)]

# 3. Extract 3x3 pixel window mean around each sampling station
print(f"Extracting 3x3 macro-pixels for {len(df_lab)} stations...")
extracted_rows = []

for _, row in df_lab.iterrows():
    lat_p, lon_p = row['Lat'], row['Lon']
    dist = (lat_sat - lat_p)**2 + (lon_sat - lon_p)**2
    y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)

    y_min, y_max = max(0, y_idx - 1), min(ds.sizes['y'], y_idx + 2)
    x_min, x_max = max(0, x_idx - 1), min(ds.sizes['x'], x_idx + 2)

    point_data = row.to_dict()
    for var in acolite_vars:
        window = ds[var].values[y_min:y_max, x_min:x_max]
        sat_mean = np.nanmean(window)

        # Convert Secchi disk depth (zsd) from meters to cm to match lab units
        if 'zsd' in var.lower():
            sat_mean = sat_mean * 100

        point_data[f"Sat_{var}"] = sat_mean

    extracted_rows.append(point_data)

df_final = pd.DataFrame(extracted_rows)
df_final.to_csv(os.path.join(output_dir, f"Extracted_Data_{base_name}.csv"), sep=';', decimal=',', index=False)

# 4. Individual validation plots (spatial comparison + linear regression)
def plot_individual_validation(df, col_lab, col_sat, param_name, unit):
    mask = ~np.isnan(df[col_lab]) & ~np.isnan(df[col_sat])
    df_valid = df[mask].sort_values(by='Punto_ID')
    if len(df_valid) < 2:
        return

    x_lab = df_valid[col_lab].values
    y_sat = df_valid[col_sat].values
    station_ids = df_valid['Punto_ID'].astype(int).astype(str).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: station-by-station comparison
    ax1.plot(station_ids, x_lab, marker='s', markersize=8, linestyle='--', color='forestgreen', label='In situ (Lab)')
    ax1.plot(station_ids, y_sat, marker='o', markersize=8, linestyle='-', color='royalblue', label='Sentinel-2 (ACOLITE)')
    ax1.set_xlabel('Sampling Station (ID)', fontsize=12)
    ax1.set_ylabel(f'{param_name} ({unit})', fontsize=12, fontweight='bold')
    ax1.set_title("Spatial Comparison", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if "Secchi" in param_name:
        ax1.invert_yaxis()

    # Right panel: scatter plot and validation metrics
    ax2.scatter(x_lab, y_sat, color='darkslateblue', s=80, alpha=0.8, edgecolor='black')
    if np.ptp(x_lab) > 0:
        slope, intercept, r_value, _, _ = linregress(x_lab, y_sat)
        n = len(x_lab)
        rmse = np.sqrt(np.mean((y_sat - x_lab)**2))
        mae = np.mean(np.abs(y_sat - x_lab))
        medae = np.median(np.abs(y_sat - x_lab))

        ax2.plot(x_lab, slope * x_lab + intercept, color='firebrick', lw=2, linestyle='--')
        stats_text = (f"y = {slope:.4f}x + {intercept:.4f}\n"
                      f"$R^2$ = {r_value**2:.4f}\n"
                      f"n = {n}\n"
                      f"RMSE = {rmse:.3f}\n"
                      f"MAE = {mae:.3f} {unit}\n"
                      f"MedAE = {medae:.3f} {unit}")
    else:
        stats_text = "Regression unavailable\n(Identical in situ values)"

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))

    ax2.set_xlabel(f'In situ / Lab ({unit})', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Sentinel-2 ACOLITE ({unit})', fontsize=12, fontweight='bold')
    ax2.set_title("Correlation & Error Metrics", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if "Secchi" in param_name:
        ax2.invert_yaxis()
        ax2.invert_xaxis()

    plt.suptitle(f"VALIDATION: {param_name.upper()}", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    clean_name = col_sat.replace('Sat_', '')
    plt.savefig(os.path.join(output_dir, f"1_Individual_{clean_name}.png"), dpi=500)
    plt.close()

print("Generating individual validation plots...")
for sat_var in acolite_vars:
    col_sat = f"Sat_{sat_var}"
    if col_sat not in df_final.columns:
        continue
    var_lower = sat_var.lower()

    if 'chl' in var_lower and 'Clorofila_LAB' in df_final.columns:
        plot_individual_validation(df_final, 'Clorofila_LAB', col_sat, f'Chlorophyll-a ({sat_var})', 'µg/L')
    elif 'spm' in var_lower and 'SS_LAB' in df_final.columns:
        plot_individual_validation(df_final, 'SS_LAB', col_sat, f'SPM ({sat_var})', 'mg/L')
    elif 'tur' in var_lower and 'TUR_LAB' in df_final.columns:
        plot_individual_validation(df_final, 'TUR_LAB', col_sat, f'Turbidity ({sat_var})', 'NTU')
    elif 'zsd' in var_lower and 'Secchi_LAB' in df_final.columns:
        plot_individual_validation(df_final, 'Secchi_LAB', col_sat, f'Secchi Disk Depth ({sat_var})', 'cm')

# 5. Multi-algorithm comparison plots (all algorithms vs. in situ)
def plot_multi_algorithm_comparison(df, col_lab, sat_prefix, param_name, unit):
    if col_lab not in df.columns:
        return

    sat_cols = [c for c in df.columns if c.lower().startswith("sat_") and sat_prefix in c.lower()]
    if not sat_cols:
        return

    df_valid = df.dropna(subset=[col_lab]).sort_values(by='Punto_ID')
    if len(df_valid) == 0:
        return

    plt.figure(figsize=(14, 7))
    station_ids_lab = df_valid['Punto_ID'].astype(int).astype(str).values

    plt.plot(station_ids_lab, df_valid[col_lab], marker='s', markersize=10, linestyle='-',
             color='black', linewidth=4, label=f'IN SITU ({unit})', zorder=10)

    for col in sat_cols:
        clean_name = col.replace('Sat_', '')
        mask_sat = ~df_valid[col].isna()
        x_pts = df_valid['Punto_ID'][mask_sat].astype(int).astype(str).values
        y_pts = df_valid[col][mask_sat].values

        if len(y_pts) > 0:
            plt.plot(x_pts, y_pts, marker='o', markersize=6, linestyle='--', linewidth=2, alpha=0.8, label=clean_name)

    plt.title(f'Multi-Algorithm Comparison: {param_name.upper()} (In Situ vs. ACOLITE)', fontsize=16, fontweight='bold')
    plt.xlabel('Sampling Station (ID)', fontsize=12, fontweight='bold')
    plt.ylabel(f'{param_name} ({unit})', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.grid(True, alpha=0.4)

    if "Secchi" in param_name:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"2_Global_Comparison_{param_name.replace(' ', '_')}.png"), dpi=500)
    plt.close()

print("Generating multi-algorithm comparison plots...")
plot_multi_algorithm_comparison(df_final, 'Clorofila_LAB', 'chl', 'Chlorophyll-a', 'µg/L')
plot_multi_algorithm_comparison(df_final, 'TUR_LAB', 'tur', 'Turbidity', 'NTU')
plot_multi_algorithm_comparison(df_final, 'SS_LAB', 'spm', 'SPM', 'mg/L')
plot_multi_algorithm_comparison(df_final, 'Secchi_LAB', 'zsd', 'Secchi Disk Depth', 'cm')

print(f"Done. Outputs saved to: {output_dir}")
print("Available NetCDF variables:", list(ds.variables))
print("Filtered ACOLITE variables:", acolite_vars)
