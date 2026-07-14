# Coastal Wetland Blue Carbon

Research code for machine-learning analysis, spatial prediction, mapping, and uncertainty assessment of deep soil organic carbon (SOC) in global unvegetated intertidal flats.

## Overview

This repository contains standalone Python scripts used to:

1. select environmental predictors and compare regression models;
2. train and serialize a LightGBM SOC model;
3. generate global, continental, and regional SOC maps;
4. visualize geographic-distance distributions for k-nearest-neighbor distance matching (k-NNDM); and
5. summarize ensemble-model uncertainty by continent for Supplementary Table 9.

The scripts are research snapshots rather than an installable Python package. Large input datasets, trained models, intermediate caches, and generated figures are not included.

## Repository structure

| Path | Purpose |
| --- | --- |
| `train_and_evaluate_soc_models.py` | Filters training data, constructs tidal proxies, performs LightGBM grid search and RFECV, compares four regressors, creates diagnostic/SHAP figures, and saves the selected model and scaler. |
| `plot_asia_soc_and_carbon_density.py` | Produces sampled point maps of predicted SOC content and carbon density across Asia. |
| `plot_knndm_distance_distributions.py` | Produces the Supplementary Fig. 18 k-NNDM distance-density illustration from reproducible seeded distributions. |
| `calculate_supplementary_table9_uncertainty_by_continent.py` | Calculates area-weighted ensemble uncertainty statistics for six continents and exports CSV, XLSX, PNG, DOCX, and JSON outputs. |
| `soc_mapping/plot_global_soc_map.py` | Predicts and grids global SOC content, then renders a Robinson-projection map. |
| `soc_mapping/plot_europe_soc_map.py` | Creates a point-based SOC prediction map for Europe. |
| `soc_mapping/plot_amazon_delta_soc_map.py` | Creates a unified-scale SOC map for the Amazon Delta. |
| `soc_mapping/plot_western_australia_soc_map.py` | Creates a unified-scale SOC map for Western Australia. |
| `soc_mapping/plot_us_east_coast_soc_map.py` | Creates a unified-scale SOC map for the United States East Coast. |
| `soc_mapping/plot_southeast_asia_soc_map.py` | Creates a unified-scale SOC map for the Southeast Asian archipelago. |
| `soc_mapping/plot_wadden_sea_soc_map.py` | Creates a unified-scale SOC map for the Wadden Sea and North Sea. |
| `soc_mapping/plot_china_yellow_sea_soc_map.py` | Creates a unified-scale SOC map for China's Yellow Sea coast. |

## Requirements

Python 3.10 or later is recommended. Install the packages required across all workflows:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy joblib scikit-learn lightgbm xgboost shap cartopy openpyxl python-docx
```

`openpyxl` and `python-docx` are only needed for the XLSX and DOCX uncertainty exports. Cartopy may download Natural Earth map assets when a requested resolution is not already cached.

## Data and model inputs

The repository does not distribute the source data or trained model files. Before running a script, update its path constants to match your environment.

The model-training workflow expects a table containing the SOC response and environmental predictors defined in `initial_features`. The mapping workflows expect `longitude`, `latitude`, and these 15 selected predictors:

```text
ndvi
BIO1_Annual_Temp
Salinity_psu
tsm_p
BIO14_Precip_Dry_Month
Vapr_kPa
sst_k
tidal_range_m
soil_moisture
bdod_1m
clay_1m
LSWI
Night_Light
BIO4_Temp_Seasonality
BIO6_Min_Temp_Coldest
```

The mapping scripts load `15个指标best_lgbm_model_refined.pkl`. The uncertainty workflow loads `ensemble_30_models_NoLog.pkl` and the six continent-specific standardized CSV files declared in `CONTINENT_FILES`.

## Path configuration

The scripts currently contain absolute Windows paths rooted at:

```text
D:\PyCharm2020(64bit)\pythonProject
```

Edit the path constants near the top of each script before execution. Depending on the workflow, these include `file_path`, `folder_path`, `data_path`, `model_path`, `save_directory`, `BASE`, `OUT_DIR`, and `DATA_DIR`.

## Usage

Run scripts independently from the repository root. Examples:

```bash
python train_and_evaluate_soc_models.py
python plot_asia_soc_and_carbon_density.py
python plot_knndm_distance_distributions.py
python soc_mapping/plot_global_soc_map.py
python soc_mapping/plot_amazon_delta_soc_map.py
python calculate_supplementary_table9_uncertainty_by_continent.py
```

Model training, global prediction, and uncertainty calculation can be memory- and compute-intensive because they process large tables and multiple models.

## Main outputs

| Workflow | Outputs |
| --- | --- |
| Model training | Feature-correlation and importance figures, RFECV curve, model-comparison plots, SHAP figures, feature-ranking XLSX, `best_lgbm_model_refined.pkl`, and `data_scaler.pkl`. |
| Asia mapping | `Asia_SOC_Map_Final.png` and `Asia_Carbon_Density_Map_Final.png`. |
| k-NNDM illustration | `k_NNDM_Geographical_Distance_Density_v6.png`. |
| Global/regional mapping | PNG maps named in each script's output configuration. |
| Continental uncertainty | Supplementary Table 9 as CSV, XLSX, PNG, and DOCX, plus JSON source notes and per-continent NPZ caches. |

## Scientific implementation notes

- Model training applies `log10(SOC + 0.01)` to the response. The global and focused regional mapping scripts convert model predictions back with `10 ** prediction - 0.01`.
- The focused regional maps share fixed SOC class boundaries: `19.9, 22.7, 25.2, 27.4, 29.8, 32.4, 35.4, 40.2, 42.9`.
- The Europe point-map script plots the model's direct predictions. Verify the intended target-transform convention before comparing it numerically with inverse-transformed maps.
- The k-NNDM plotting script uses NumPy random draws with seed 42 to create an illustrative, reproducible density figure; it does not read measured distance observations.
- The uncertainty workflow defines relative ensemble error as `SD / (mean + 0.05)`, scales the global 2nd–98th percentile range to an NUI range of 0.05–0.50, and uses cosine-latitude area weights.

## Reproducibility limitations

- Data and trained model artifacts are external to this repository.
- Absolute local paths must be configured manually.
- Several plotting scripts suppress warnings or skip malformed input files, so inspect source-data quality before production use.
- End-to-end results depend on package versions, external Natural Earth assets, and the exact unpublished input files.

## License

Licensed under the [Apache License 2.0](LICENSE).
