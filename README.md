# Shallow-lake temperature models and hierarchical diagnostics

Code accompanying **High-Frequency Observations Reveal Trade-Offs Between Profile Accuracy and Thermal-Structure Fidelity in Small Shallow Lake Temperature Models**.

This repository provides LSTM, residual state-space (rSSM), and partially physics-guided neural network (PGNN) implementations for Yamanokami, Shiode, and Shin reservoirs. It also includes observation screening, thermal-structure and event diagnostics, sensitivity analyses, and plotting scripts for the revised manuscript.

**GLM simulation, calibration code, configuration files, and executables are not included.** Four-model comparison workflows read separately supplied GLM predictions or summary statistics.

## 1. Installation

Use Python 3.10 or later. Run the commands below from the repository root:

```bash
python -m pip install -r requirements.txt
python run.py list
```

The required packages are NumPy, pandas, Matplotlib, PyTorch, and tqdm. Training on a GPU requires a compatible CUDA-enabled PyTorch installation. For available arguments, use:

```bash
python run.py cross-year --help
python run.py fig08 --help
```

If the `tests/` directory is included, run the local checks with:

```bash
python -m unittest discover -s tests -v
```

These checks test code integrity and small synthetic examples; they do not reproduce the full training experiments.

## 2. Repository structure

```text
.
├── training/        Main cross-year experiments and revised Shin training
├── src/             Shared data-loading, dataset, model, and training modules
├── configs/         Workflow registry, final member lists, and data-path metadata
├── analysis/        Observation screening, metrics, events, and stability diagnostics
├── sensitivity/     Additional chronological-split experiments
├── figures/         Final numerical-figure workflows
├── data/            Location for separately supplied research data
├── tests/           Local integrity and synthetic tests
├── run.py           Workflow dispatcher
├── requirements.txt
├── LICENSE
└── README.md
```

Keep the directory structure intact. `run.py` reads `configs/entrypoints.json`, and the cross-year training driver reads `configs/final_members.json`. Experiment-specific files named `models.py`, `settings.py`, and `engine.py` are separate implementations and should not be merged.

## 3. Data availability and placement

**Data and experimental results (Zenodo): [10.5281/zenodo.24185850](https://doi.org/10.5281/zenodo.24185850).**

Download the data and result packages from this record and place them in the directories specified below. The code is maintained separately in this GitHub repository.

Observations, model predictions, checkpoints, and figure-source tables are managed separately from the code. Full training or reanalysis requires the corresponding input data. Figure-source tables alone support selected plotting workflows, not complete model reproduction.

Use the following layout:

```text
data/
├── observations/          Aligned meteorological and temperature observations
├── archives/              Final event and member-prediction archives
├── figure_source_data/    Final figure-source tables, grouped by figure
└── checkpoints/           Externally supplied model weights, when available
```

The training workflows use these observation files in `data/observations/`:

```text
yamanokami_2024_aligned_qc_wide.csv
yamanokami_2025_aligned_qc_wide.csv
shiode_2024_aligned_qc_wide.csv
shiode_2025_aligned_qc_wide.csv
shin_2025_aligned_qc_wide.csv
```

The wide-format input contains `time`; meteorological columns `solar`, `precip`, `wind`, `air_temp`, `rh`, and `pressure`; depth-labeled temperature columns such as `0.25m`; and the associated quality-control fields required by each workflow, including `<depth>_qc_bad`, `<depth>_flag_nan`, `<depth>_flag_range`, `<depth>_flag_spike`, and `<depth>_flag_stuck`. Preserve the supplied depth labels, time axes, and quality flags.

Figure 2 uses the corresponding five `*_aligned_qc_long.csv` files, with `time`, `depth_m`, `temp`, `flag_nan`, `flag_range`, `flag_spike`, and `flag_stuck`. Meteorological composites and stability analyses additionally use `yamanokami_2024_aligned.csv` and `shiode_2025_aligned.csv`.

For the figure-source companion archive, copy its **inner `figure_source_data/` directory** into `data/`. For example, Figure 8 should be located at:

```text
data/figure_source_data/figure08/Figure8_event_level_drop_ratios.csv
```

## 4. Training and prediction

### Main cross-year experiments

Yamanokami uses 2025 for model development and 2024 for testing. Shiode uses 2024 for development and 2025 for testing. After supplying the observation files, run:

```bash
python run.py cross-year --site yamanokami --model pgnn --phase train --data-dir data/observations --out outputs/yamanokami/pgnn
python run.py cross-year --site yamanokami --model pgnn --phase predict --data-dir data/observations --out outputs/yamanokami/pgnn
```

Supported sites are `yamanokami` and `shiode`; supported models are `lstm`, `rssm`, and `pgnn`. The driver uses the final member IDs in `configs/final_members.json`. Prediction requires the trained checkpoints in the same output directory. Existing training checkpoints are protected unless `--overwrite` is specified.

### Main Shin experiment

Shin uses the revised chronological split: 56% training, 14% validation, and 30% testing.

```bash
python run.py shin-lstm --csv data/observations/shin_2025_aligned_qc_wide.csv --out outputs/shin/lstm --phase train --device cuda:0
python run.py shin-lstm --csv data/observations/shin_2025_aligned_qc_wide.csv --out outputs/shin/lstm --phase evaluate --device cuda:0
```

Use `shin-rssm` or `shin-pgnn` for the other models, with a separate output directory for each. Use `--device cpu` when CUDA is unavailable.

### Additional chronological-split experiments

The Table B3 sensitivity experiments use separate workflows and do not replace the main cross-year experiments:

```bash
python run.py yamanokami-split-pgnn --csv data/observations/yamanokami_2025_aligned_qc_wide.csv --out outputs/sensitivity/yamanokami/pgnn --phase train --device cuda:0
```

The available task names are `yamanokami-split-lstm`, `yamanokami-split-rssm`, `yamanokami-split-pgnn`, and the corresponding `shiode-split-*` tasks. Shiode uses the 2024 observation file. Run the same task with `--phase evaluate` after training.

## 5. Analyses and figures

A typical analysis order is observation screening and event detection, evaluation of member predictions, event-magnitude and threshold analyses, meteorological/stability diagnostics, and plotting. For example:

```bash
python run.py events-shiode --root data --obs-csv data/observations/shiode_2025_aligned_qc_wide.csv --out-root outputs/events/shiode
python run.py events-yamanokami-shin --help
python run.py fig08-rebuild --help
python run.py figC2-rebuild --help
python run.py stability --help
```

The Yamanokami/Shin event workflow reads the output of the `observations-yamanokami-shin` completeness check. Use each task's `--help` for its input paths. The final event scores are calculated by `fig08-rebuild` and `figC2-rebuild`: compute the largest temporally ordered decrease for each member, then average member-level ratios within each event. Do not substitute older ensemble-first or unordered-range scores.

### Replot from the final source tables

```bash
python run.py fig05 --csv data/figure_source_data/figure05/figure5_layerwise_summary.csv --outdir outputs/figures/figure05
python run.py fig08 --event-csv data/figure_source_data/figure08/Figure8_event_level_drop_ratios.csv --outdir outputs/figures/figure08
```

| Paper item | Task name | Main input |
|---|---|---|
| Figure 2 | `fig02` | Five QC-long observation files |
| Figure 4 / Table 2 | `fig04` | `all_sites_current_profile_metrics.csv` |
| Figure 5 | `fig05` | `figure5_layerwise_summary.csv` |
| Figure 6 / Table 3 | `fig06` | `figure6_structure_metrics.csv` |
| Figure 7 | `fig07` | Final event and member-prediction archives |
| Figure 8 | `fig08` | `Figure8_event_level_drop_ratios.csv` |
| Figure 9 | `fig09` | Event archives and meteorological inputs |
| Figure C1 | `figC1` | `threshold_event_counts.csv` |
| Figure C2 | `figC2` | `FigureC2_event_level_ratios.csv` |
| Figure C3 | `figC3` | Event, observation, and member-prediction archives |
| Figure D1 | `figD1` | Final stability and stability-summary CSVs |
| Figure D2 | `figD2` | Final stability CSV |
| Table B3 | `yamanokami-split-*`, `shiode-split-*` | Chronological-split training and evaluation |
| Table B4 | `split-context` | Temporal coverage and event-frequency inputs |

Figures 2, 7, 9, and C3 require upstream observations or result archives in the current implementations; the figure-source CSV companion alone cannot run those workflows. Figures 4 and D2 use standalone exporters of final result tables; the exporters do not recompute the scientific metrics. Figures 1 and 3 are not included as executable numerical workflows.

## 6. Model and evaluation scope

PGNN uses an LSTM with a temperature-range penalty and losses on vertical temperature standard deviation (STD) and mean absolute vertical temperature gradient (Gmean). These are physically motivated, observation-based structural constraints. They do not enforce energy conservation or explicitly represent mixing dynamics.

Training losses and fixed-depth evaluation use different sample-completeness rules. Event detection requires complete, quality-screened observations across the selected depths throughout each 3-hour window. Drop ratios evaluate event magnitudes within observation-defined windows; they do not measure independent event-occurrence or timing skill.

## 7. License and citation

The code is released under the **MIT License**; see [LICENSE](LICENSE). The license for the separately archived data is specified in the [Zenodo record](https://doi.org/10.5281/zenodo.24185850).

Please cite the accompanying paper, when available, and the dataset: **[10.5281/zenodo.24185850](https://doi.org/10.5281/zenodo.24185850)**. Use the citation metadata provided by Zenodo for the dataset.

This DOI identifies the companion data record, not an archived version of this code repository. When referring to the code, specify the GitHub repository and the release tag or commit used. Final publication details will be added when available.
