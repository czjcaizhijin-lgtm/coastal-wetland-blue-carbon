# Standardize Script Names and Add Project README — Design

## Objective

Publish the existing coastal wetland soil organic carbon (SOC) research scripts with clear English filenames and an English README, without changing scientific calculations, model behavior, path constants, plot settings, or output values.

## Repository Location

All final work is performed directly in `D:\PyCharm2020(64bit)\pythonProject\github`. The folder is initialized as the working copy of `https://github.com/czjcaizhijin-lgtm/coastal-wetland-blue-carbon.git`, on `main`, tracking `origin/main`.

## Chosen Approach

Preserve the existing root layout and one mapping subdirectory. Rename only Chinese or unclear file and directory names to descriptive lowercase `snake_case` English names. Keep the already-compliant uncertainty filename unchanged. Do not reorganize the workflow or consolidate the near-duplicate mapping scripts.

## Filename Mapping

| Source path | Repository path |
| --- | --- |
| `15个指标筛选.py` | `train_and_evaluate_soc_models.py` |
| `筛选后的反演.py` | `plot_asia_soc_and_carbon_density.py` |
| `K-NNDM2.py` | `plot_knndm_distance_distributions.py` |
| `calculate_supplementary_table9_uncertainty_by_continent.py` | `calculate_supplementary_table9_uncertainty_by_continent.py` |
| `反演代码/0425定全球-像素格内平均值-网格填充加粗处理.py` | `soc_mapping/plot_global_soc_map.py` |
| `反演代码/澳大利亚西海岸放大.py` | `soc_mapping/plot_western_australia_soc_map.py` |
| `反演代码/北美东海岸放大.py` | `soc_mapping/plot_us_east_coast_soc_map.py` |
| `反演代码/东南亚群岛放大.py` | `soc_mapping/plot_southeast_asia_soc_map.py` |
| `反演代码/欧洲北海与瓦登海放大.py` | `soc_mapping/plot_wadden_sea_soc_map.py` |
| `反演代码/欧洲反演.py` | `soc_mapping/plot_europe_soc_map.py` |
| `反演代码/亚马逊放大图.py` | `soc_mapping/plot_amazon_delta_soc_map.py` |
| `反演代码/中国黄海沿岸放大.py` | `soc_mapping/plot_china_yellow_sea_soc_map.py` |

## README Scope

The English `README.md` documents the project purpose, workflow, every script, Python dependencies, expected data/model inputs, the 15 mapping features, path configuration, representative commands, outputs, scientific implementation notes, reproducibility limitations, and Apache-2.0 license.

It explicitly states that datasets and trained models are not included, paths are machine-specific, the k-NNDM figure uses seeded simulated distributions, and target-transform conventions should be checked before comparing the Europe map with the inverse-transformed global and local maps.

## Validation

- Record each source script's SHA-256 hash before renaming and compare it with the target hash after renaming.
- Confirm all 12 approved Python paths exist.
- Confirm Python filenames and repository directory names contain only portable English naming characters.
- Parse all Python files with `ast.parse` without importing optional packages or producing bytecode.
- Verify every local Markdown path in the README exists.
- Run `git diff --check`, inspect the staged diff, and confirm only approved files are staged.

The scientific workflows cannot be executed end to end because their large datasets, model artifacts, and map assets are not part of the repository.

## Publication

Commit the approved documentation and implementation on `main`, then push directly to `origin/main` as requested. Do not create a pull request.

## Out of Scope

- Renaming Python variables, functions, data columns, feature columns, or generated outputs.
- Replacing absolute paths with configuration or command-line arguments.
- Changing numerical, scientific, plotting, or error-handling behavior.
- Deduplicating scripts or adding datasets, models, figures, or dependency lock files.
