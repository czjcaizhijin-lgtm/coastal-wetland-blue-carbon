# Standardize Script Names and Add README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the 12 research scripts to portable English paths, add accurate project documentation, and publish the result to the repository's `main` branch.

**Architecture:** Preserve every script byte-for-byte and change only filesystem paths. Add a single root README that explains the existing independent research workflows and their external data/model requirements.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, LightGBM, XGBoost, SHAP, Cartopy, Joblib, OpenPyXL, python-docx, Git.

## Global Constraints

- Work directly in `D:\PyCharm2020(64bit)\pythonProject\github`.
- Keep all scientific source bytes unchanged; only file and directory paths may change.
- Do not add datasets, trained models, generated figures, or dependency lock files.
- Keep machine-specific data paths unchanged and document them.
- Commit and push directly to `origin/main`; do not create a pull request.

---

### Task 1: Rename Python scripts without changing content

**Files:**
- Rename: `15个指标筛选.py` → `train_and_evaluate_soc_models.py`
- Rename: `筛选后的反演.py` → `plot_asia_soc_and_carbon_density.py`
- Rename: `K-NNDM2.py` → `plot_knndm_distance_distributions.py`
- Keep: `calculate_supplementary_table9_uncertainty_by_continent.py`
- Rename directory: `反演代码` → `soc_mapping`
- Rename the eight mapping scripts according to the design mapping.

**Interfaces:**
- Consumes: the 12 original Python files at the paths listed in the design.
- Produces: the same 12 byte streams at the approved English paths.

- [ ] **Step 1: Capture source hashes**

Use `Get-FileHash -Algorithm SHA256` for every source path and retain the path-to-hash mapping in memory for the duration of the rename command.

- [ ] **Step 2: Rename with literal paths**

Use PowerShell `Move-Item -LiteralPath` for each file and the mapping directory. Fail before moving if a source is missing or target already exists.

- [ ] **Step 3: Verify content identity**

Recalculate each target SHA-256 hash and compare it with the corresponding source hash captured in Step 1. Abort if any value differs.

- [ ] **Step 4: Verify the resulting paths**

Run:

```powershell
rg --files -g '*.py'
```

Expected: exactly 12 Python files, with the paths defined in the design and no Chinese characters, spaces, or hyphens in Python paths.

---

### Task 2: Add the English project README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the approved English script paths and the behavior observed in the 12 source scripts.
- Produces: user-facing setup, workflow, usage, and reproducibility documentation.

- [ ] **Step 1: Create `README.md` with the approved content**

```markdown
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
```

- [ ] **Step 2: Verify README path references**

Extract all local Markdown code paths ending in `.py` or `LICENSE` and verify that each path exists in the repository.

---

### Task 3: Validate, commit, and publish

**Files:**
- Validate: all 12 Python scripts and `README.md`
- Track: `docs/superpowers/specs/2026-07-14-standardize-script-names-design.md`
- Track: `docs/superpowers/plans/2026-07-14-standardize-script-names.md`

**Interfaces:**
- Consumes: the renamed source tree and README.
- Produces: a verified commit on `main` and a matching `origin/main`.

- [ ] **Step 1: Parse all Python sources without importing dependencies**

Run an in-memory `ast.parse` over every `*.py` file. Expected: 12 files parsed, zero syntax errors, and no `__pycache__` output.

- [ ] **Step 2: Check names and Git whitespace**

Run a portable-name scan and `git diff --check`. Expected: no invalid Python paths and no whitespace errors.

- [ ] **Step 3: Inspect and stage only approved files**

Run `git status -sb`, inspect the full diff, and stage `README.md`, the 12 Python scripts, `soc_mapping`, and the two Superpowers documents. Confirm `git diff --cached --name-status` contains no unrelated files.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: publish coastal wetland SOC research scripts"
```

- [ ] **Step 5: Push and verify remote equality**

```bash
git push origin main
git fetch origin main
git rev-parse HEAD
git rev-parse origin/main
```

Expected: push succeeds and the two commit hashes are identical.
