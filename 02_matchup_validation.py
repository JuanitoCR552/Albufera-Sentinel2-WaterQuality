import os
import glob
import subprocess

# 1. File paths (replace with your local directories)
input_safe_dir = "PUT_PATH_TO_YOUR_SENTINEL2_SAFE_FOLDER"
output_nc_dir = "PUT_PATH_TO_YOUR_OUTPUT_NETCDF_FOLDER"
acolite_script_path = "PUT_PATH_TO_YOUR_ACOLITE_FOLDER/launch_acolite.py"

os.makedirs(output_nc_dir, exist_ok=True)

# 2. ACOLITE configuration (L'Albufera bounding box + L2W bio-optical algorithms)
l2w_parameters = (
    "chl_re_mishra,chl_re_gons740,"
    "spm_nechad2016_865,spm_nechad2016_665,"
    "tur_nechad2016_665,tur_nechad2016_865,"
    "p3qaa_zSD,par,kd_lee"
)

# limit = South, West, North, East
settings_base = f"""limit=39.15,-0.50,39.50,-0.15
atmospheric_correction_method=dark_spectrum
l2w_parameters={l2w_parameters}
rgb_rhot=False
rgb_rhos=False
map_l2w=False
l2r_delete_netcdf=True
l1r_delete_netcdf=True
s2_target_res=10
l2w_mask_threshold=0.05
l2w_mask_negative_rhow=True
output_geolocation=True
netcdf_projection=True
atmospheric_correction=True
tact_run=False
"""

safe_folders = sorted(glob.glob(os.path.join(input_safe_dir, "*.SAFE")))

if not safe_folders:
    print("No .SAFE folders found in the input directory.")
    exit()

print(f"Found {len(safe_folders)} Sentinel-2 scenes to process.")

# 3. Batch processing loop (skips already processed dates)
temp_settings_file = os.path.join(output_nc_dir, "temp_acolite_settings.txt")

for i, safe_path in enumerate(safe_folders, 1):
    safe_name = os.path.basename(safe_path)
    scene_id = safe_name.replace(".SAFE", "")

    existing_files = glob.glob(os.path.join(output_nc_dir, f"*{scene_id}*L2W.nc"))
    if existing_files:
        print(f"[{i}/{len(safe_folders)}] Skipping {safe_name} (already processed)")
        continue

    print(f"[{i}/{len(safe_folders)}] Processing {safe_name}...")

    with open(temp_settings_file, "w") as f:
        f.write(f"inputfile={safe_path}\n")
        f.write(f"output={output_nc_dir}\n")
        f.write(settings_base)

    try:
        subprocess.run(["python", acolite_script_path, "--cli", "--settings", temp_settings_file], check=True)
        print(f"Completed: {safe_name}")
    except subprocess.CalledProcessError:
        print(f"Error processing {safe_name}. Skipping to next scene...")

if os.path.exists(temp_settings_file):
    os.remove(temp_settings_file)

print("Batch processing finished.")
