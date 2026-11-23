# Sentinel-2 Water Quality Retrieval & Monitoring in L'Albufera de València (2023–2026)

Python workflow and bio-optical processing scripts associated with the research article published in *Marine Pollution Bulletin*:

> **From chronic hyper-eutrophication to extreme storm discharges: Multi-year satellite monitoring of water quality in L'Albufera de València**  
> Juan Esteban Caicedo-Rodríguez, Mar Roca-Mora, Ana María Blázquez-Morilla  
> *Marine Pollution Bulletin*, 120608 (2026)  
> **DOI:** [https://doi.org/10.1016/j.marpolbul.2026.120608](https://doi.org/10.1016/j.marpolbul.2026.120608)

---

## Authors & Affiliations

* **Juan Esteban Caicedo-Rodríguez** *(Corresponding Author)* — GEOCOAST Research Group, Environmental and Marine Sciences Research Institute (IMEDMAR-UCV), Catholic University of Valencia, C/ Guillem de Castro 94, 46001 Valencia, Spain.  
  *Contact:* `juanesteban.caicedo@ucv.es`
* **Mar Roca-Mora** — Ecology and Coastal Management Department, Institute of Marine Sciences of Andalusia (ICMAN), Spanish National Research Council (CSIC), Av. República Saharaui, s/n, 11519 Puerto Real, Cádiz, Spain.
* **Ana María Blázquez-Morilla** — GEOCOAST Research Group, Environmental and Marine Sciences Research Institute (IMEDMAR-UCV), Catholic University of Valencia, C/ Guillem de Castro 94, 46001 Valencia, Spain.

---

## Overview

This repository provides the end-to-end Python pipeline used to monitor spatio-temporal water quality and optical dynamics in **L'Albufera de València** (eastern Iberian Peninsula) from **June 2023 to May 2026** using Sentinel-2 (A/B/C) MSI imagery. The workflow covers:

1. Automated atmospheric correction and aquatic parameter retrieval via **ACOLITE** (Dark Spectrum Fitting, 10 m spatial resolution).
2. Satellite–*in situ* match-up validation (3×3 pixel windows) across five field campaigns (2025–2026) for Chlorophyll-a (`chl_re_mishra`), Suspended Particulate Matter (`spm_nechad2016_665` / `865`), Turbidity (`tur_nechad2016_665` / `865`, `tur_dogliotti2015`), Secchi Disk Depth (`p3qaa_zSD`), Euphotic Depth (`qaa_v6_Zeu_Lee`), and Diffuse Attenuation Coefficient (`qaa_v6_KdPAR_Nechad`).
3. Multi-year time-series reconstruction, seasonal spatial composites, inter-parameter correlation analyses, and quantitative anomaly mapping of the **October 29, 2024 DANA** extreme flash-flood event.

---

## Repository Structure

The scripts are numbered sequentially according to the methodological workflow of the study:

| Script | Description |
| :--- | :--- |
| `01_acolite_batch_processing.py` | Batch atmospheric correction (DSF) and Level-2 Water (L2W) parameter retrieval from Sentinel-2 `.SAFE` scenes using ACOLITE CLI. |
| `02_matchup_validation.py` | Extracts 3×3 macro-pixel means around sampling stations and generates campaign-level spatial comparisons and linear regression plots. |
| `03_global_validation_metrics.py` | Pools all match-up campaigns to compute global validation metrics (N, R², *p*-value, Slope, Intercept, RMSE, RE, Bias, MAE, MedAE). |
| `04_water_quality_time_series.py` | Reconstructs the 2023–2026 lagoon-wide time series with ±MAE / ±SD uncertainty bands and compares NIR vs. red-band stability. |
| `05_dana_impact_statistics.py` | Computes spatial mean and standard deviation (±SD) of SPM comparing the pre-event baseline (September 2024) vs. post-DANA conditions (November 2024). |
| `06_dana_impact_metrics.py` | Quantifies lagoon-wide pre/post-DANA means, net variations, and percentage changes across all six bio-optical and optical parameters. |
| `07_dana_impact_maps.py` | Generates three-panel spatial anomaly maps (*Baseline*, *Post-Event*, and *Impact Anomaly*) for the October 2024 DANA event. |
| `08_seasonal_spatial_maps.py` | Aggregates valid Sentinel-2 scenes into meteorological seasonal composites (2023–2026) and exports seasonal lagoon-wide summary tables. |
| `09_interparameter_correlations.py` | Performs outlier filtering (3×IQR) and generates pairwise scatter plots and linear regressions among bio-optical variables. |

---

## Requirements & Usage

### Dependencies
* Python 3.9+
* Core packages: `numpy`, `pandas`, `xarray`, `geopandas`, `rioxarray`, `matplotlib`, `scipy`
* [ACOLITE](https://github.com/acolite/acolite) (tested with Python version `20251013.0`) for Level-1C to Level-2W processing.

### Running the Scripts
All scripts use explicit path placeholders (`PUT_PATH_TO_...`) at the top of each file. Before running any script, replace those strings with your local directories containing the Sentinel-2 `.SAFE` scenes, ACOLITE `L2W.nc` outputs, *in situ* Excel spreadsheets, and the L'Albufera boundary shapefile.

### Acknowledgements
This work was supported by the Generalitat Valenciana (project CIAICO/2024/97), the Spanish Ministry of Science, Innovation and Universities (Grant FPU20/01294), and the Universidad Católica de Valencia (project UCV-2024-153-902).

---

## How to Cite

If you use these scripts or workflows in your research, please cite our article:

> Caicedo-Rodríguez, J.E., Roca-Mora, M., Blázquez-Morilla, A.M., 2026. From chronic hyper-eutrophication to extreme storm discharges: Multi-year satellite monitoring of water quality in L'Albufera de València. *Mar. Pollut. Bull.* 120608. https://doi.org/10.1016/j.marpolbul.2026.120608

**BibTeX:**

```bibtex
@article{CaicedoRodriguez2026Albufera,
  title = {From chronic hyper-eutrophication to extreme storm discharges: Multi-year satellite monitoring of water quality in L'Albufera de València},
  author = {Caicedo-Rodríguez, Juan Esteban and Roca-Mora, Mar and Blázquez-Morilla, Ana María},
  journal = {Marine Pollution Bulletin},
  pages = {120608},
  year = {2026},
  doi = {10.1016/j.marpolbul.2026.120608}
}
