import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

# 1. Input campaigns and output path (replace with your local directories)
campaigns = [
    {
        "date": "Campaign_1",
        "excel": "PUT_PATH_TO_CAMPAIGN_1_IN_SITU_EXCEL.xlsx",
        "nc": "PUT_PATH_TO_CAMPAIGN_1_ACOLITE_L2W.nc"
    },
    {
        "date": "Campaign_2",
        "excel": "PUT_PATH_TO_CAMPAIGN_2_IN_SITU_EXCEL.xlsx",
        "nc": "PUT_PATH_TO_CAMPAIGN_2_ACOLITE_L2W.nc"
    },
    {
        "date": "Campaign_3",
        "excel": "PUT_PATH_TO_CAMPAIGN_3_IN_SITU_EXCEL.xlsx",
        "nc": "PUT_PATH_TO_CAMPAIGN_3_ACOLITE_L2W.nc"
    },
    {
        "date": "Campaign_4",
        "excel": "PUT_PATH_TO_CAMPAIGN_4_IN_SITU_EXCEL.xlsx",
        "nc": "PUT_PATH_TO_CAMPAIGN_4_ACOLITE_L2W.nc"
    },
    {
        "date": "Campaign_5",
        "excel": "PUT_PATH_TO_CAMPAIGN_5_IN_SITU_EXCEL.xlsx",
        "nc": "PUT_PATH_TO_CAMPAIGN_5_ACOLITE_L2W.nc"
    }
]

output_excel_path = "PUT_PATH_TO_YOUR_OUTPUT_FOLDER/Global_Validation_Metrics.xlsx"

# 2. Multi-campaign 3x3 macro-pixel extraction
# Expected Excel columns: Lat, Lon, Clorofila_LAB, SS_LAB, TUR_LAB, Secchi_LAB (or DISCO_LAB), PAR_LAB
print("Starting multi-campaign data extraction...")
pooled_records = []

for camp in campaigns:
    print(f"Processing {camp['date']}...")

    df_lab = pd.read_excel(camp['excel'], decimal=',')
    df_lab.columns = df_lab.columns.str.strip()
    df_lab['Lat'] = pd.to_numeric(df_lab['Lat'].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
    df_lab['Lon'] = pd.to_numeric(df_lab['Lon'].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
    df_lab = df_lab.dropna(subset=['Lat', 'Lon'])

    # Convert Secchi disk depth from cm to meters
    if 'DISCO_LAB' in df_lab.columns:
        df_lab['Secchi_LAB_m'] = df_lab['DISCO_LAB'] / 100.0
    elif 'Secchi_LAB' in df_lab.columns:
        df_lab['Secchi_LAB_m'] = df_lab['Secchi_LAB'] / 100.0

    ds = xr.open_dataset(camp['nc'])
    lat_sat = ds['lat'].values
    lon_sat = ds['lon'].values
    keywords = ['chl', 'spm', 'tur', 'zsd', 'kpar', 'zeu', 'fai', 'ndvi']
    acolite_vars = [v for v in ds.variables if any(k in v.lower() for k in keywords)]

    for _, row in df_lab.iterrows():
        lat_p, lon_p = row['Lat'], row['Lon']
        dist = (lat_sat - lat_p)**2 + (lon_sat - lon_p)**2
        y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)

        y_min, y_max = max(0, y_idx - 1), min(ds.sizes['y'], y_idx + 2)
        x_min, x_max = max(0, x_idx - 1), min(ds.sizes['x'], x_idx + 2)

        point_data = row.to_dict()
        for var in acolite_vars:
            window = ds[var].values[y_min:y_max, x_min:x_max]
            point_data[f"Sat_{var}"] = np.nanmean(window)

        pooled_records.append(point_data)

df_global = pd.DataFrame(pooled_records)
print(f"Extraction finished. Total pooled match-up samples: {len(df_global)}")

# 3. Compute validation metrics (N, R2, p-value, Slope, Intercept, RMSE, RE, Bias, MAE, MedAE)
print("Calculating validation metrics...")

parameter_pairs = {
    'Clorofila_LAB': {'prefix': 'chl', 'unit': 'µg/L', 'name': 'Chlorophyll-a'},
    'SS_LAB': {'prefix': 'spm', 'unit': 'mg/L', 'name': 'SPM'},
    'TUR_LAB': {'prefix': 'tur', 'unit': 'NTU', 'name': 'Turbidity'},
    'Secchi_LAB_m': {'prefix': 'zsd', 'unit': 'm', 'name': 'Secchi Disk Depth'},
    'PAR_LAB': {'prefix': 'kpar', 'unit': 'm^-1', 'name': 'K_PAR'}
}

summary_stats = []

for col_lab, info in parameter_pairs.items():
    if col_lab not in df_global.columns:
        continue

    prefix = info['prefix']
    sat_cols = [c for c in df_global.columns if c.lower().startswith("sat_") and prefix in c.lower()]

    for col_sat in sat_cols:
        df_valid = df_global.dropna(subset=[col_lab, col_sat])
        n_points = len(df_valid)

        if n_points < 3:
            continue

        x_lab = df_valid[col_lab].values
        y_sat = df_valid[col_sat].values

        if np.ptp(x_lab) == 0:
            continue

        slope, intercept, r_value, p_value, _ = linregress(x_lab, y_sat)
        r2 = r_value**2

        errors = y_sat - x_lab
        bias = np.mean(errors)
        mae = np.mean(np.abs(errors))
        medae = np.median(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        re = np.mean(np.abs(errors) / x_lab) * 100.0

        algorithm_name = col_sat.replace('Sat_', '')

        summary_stats.append({
            'Parameter': info['name'],
            'Satellite Algorithm': algorithm_name,
            'Unit': info['unit'],
            'N': n_points,
            'R2': round(r2, 4),
            'p-value': f"{p_value:.2e}",
            'Slope': round(slope, 4),
            'Intercept': round(intercept, 4),
            'RMSE': round(rmse, 4),
            'RE (%)': round(re, 2),
            'Bias': round(bias, 4),
            'MAE': round(mae, 4),
            'MedAE': round(medae, 4)
        })

# 4. Export final validation table to Excel
df_results = pd.DataFrame(summary_stats)
df_results = df_results.sort_values(by=['Parameter', 'R2'], ascending=[True, False])

os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
df_results.to_excel(output_excel_path, index=False)
print(f"Summary table saved to: {output_excel_path}")
