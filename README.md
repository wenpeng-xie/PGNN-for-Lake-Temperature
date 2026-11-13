# Shallow-lake temperature models and hierarchical diagnostics

Code accompanying **High-Frequency Observations Reveal Trade-Offs Between Profile
Accuracy and Thermal-Structure Fidelity in Small Shallow Lake Temperature Models**.

This repository contains the final source versions identified for the JoHRS revision:
LSTM, residual state-space model (rSSM), and partially physics-guided neural network
(PGNN); observation screening; thermal-structure and event diagnostics; sensitivity
analyses; and publication-figure workflows.

**Scope:** GLM simulation, calibration, namelists, and executables are not included.
Comparison workflows can read externally supplied GLM predictions/summary values;
they do not run or calibrate GLM. The old `Figure.ipynb` and `Figure-S.ipynb` are not
included. The cross-year model sources were exported from the unchanged training
cells; the revised Shin training implementation replaces its superseded workflow.

## Installation

Use a Python environment with NumPy, pandas, Matplotlib, PyTorch and tqdm:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py list
```

For plotting only, install `requirements-plotting.txt`. The recorded Shin training
environment used Python 3.10, PyTorch 2.5.1+cu121, NumPy 2.2.6 and pandas 2.3.3.
The original environment records are in `provenance/training_environment/`.
Install the PyTorch build appropriate to the target CPU/GPU environment; the
unbounded requirements file is not a claim of bitwise reproducibility on all builds.

## Data availability and directory layout

Code is intended for GitHub. Observations, checkpoints, member predictions and
figure-source tables are intended for a separate Zenodo record. **A Zenodo DOI has
not yet been assigned in this source release.** No DOI or repository address is
invented here. `configs/data_paths.json` is the place to record the final data link.

Place external data under the locations documented in [DATA_LAYOUT](docs/DATA_LAYOUT.md).
A separate small figure-source-data companion can be copied into `data/` for direct
replotting; it is not the full training/observation archive.

## Training and prediction

### Main cross-year experiments: Yamanokami and Shiode

These use the original training implementation, including its preprocessing,
losses, validation logic and model classes. The dispatcher changes input/output
paths and exposes the frozen final member list; it does not silently replace the
cross-year protocol with the matched-split sensitivity experiment.

```bash
python run.py cross-year --site yamanokami --model pgnn --phase train --data-dir data/observations --out outputs/yamanokami/pgnn
python run.py cross-year --site yamanokami --model pgnn --phase predict --data-dir data/observations --out outputs/yamanokami/pgnn
```

Use `--site shiode` and `--model lstm`, `rssm`, or `pgnn` as appropriate. The default
members are in `configs/final_members.json`. `--seeds 42` can run one member for a
local check, but an overridden seed list is not the published ensemble. Existing
checkpoints are protected unless `--overwrite` is explicitly passed.

### Main Shin experiment: revised independent chronological split

The final `Shin_ML_iso4_retrain_v1` implementation is preserved in `training/shin/`:

```bash
python run.py shin-lstm --csv data/observations/shin_2025_aligned_qc_wide.csv --out outputs/shin/lstm --phase train --device cuda:0
python run.py shin-lstm --csv data/observations/shin_2025_aligned_qc_wide.csv --out outputs/shin/lstm --phase evaluate --device cuda:0
```

Replace `shin-lstm` with `shin-rssm` or `shin-pgnn`. Use `--device cpu` where needed.
The main split is 56% training / 14% validation / 30% testing. Evaluation uses frozen
checkpoints. The source program refuses invalid split/coverage conditions.

### Additional split-sensitivity experiments (Table B3 only)

`yamanokami-split-*` and `shiode-split-*` call the separate final sensitivity packages.
They are not replacements for the cross-year experiments:

```bash
python run.py yamanokami-split-pgnn --csv data/observations/yamanokami_2025_aligned_qc_wide.csv --out outputs/sensitivity/yamanokami/pgnn --phase train --device cuda:0
```

Each workflow's `--help` lists its actual source arguments.

## Analyses and figures

```bash
python run.py events-shiode --help
python run.py events-yamanokami-shin --help
python run.py fig08-rebuild --help
python run.py figC2-rebuild --help
python run.py stability --help
python run.py fig05 --csv data/figure_source_data/figure05/figure5_layerwise_summary.csv --outdir outputs/figures/figure05
```

[FIGURE_TABLE_MAP](docs/FIGURE_TABLE_MAP.md) maps every included figure to its source
version, inputs and entry point. [WORKFLOWS](docs/WORKFLOWS.md) explains the event and
analysis dependencies. Commands run scripts in separate processes to avoid clashes
among experiment-specific files named `models.py`, `settings.py`, and `engine.py`.

## Model interpretation

PGNN keeps the LSTM architecture. Temperature-range, STD and Gmean penalties add
physically motivated, observation-based structural supervision. They are not
energy-conservation residuals or mixing-process equations. The PGNN heat/H term is
inactive in the final configurations; the final weights and member IDs are retained
from the supplied sources. Training losses and fixed-depth evaluation masks are
different parts of the workflow and must not be interchanged.

## Provenance and validation scope

`provenance/source_manifest.csv` records source and packaged hashes.
[IMPLEMENTATION_NOTES](docs/IMPLEMENTATION_NOTES.md) documents packaging-only changes,
source protocol details, and the two standalone final-table exporters added for
Figures 4 and D2. No model was retrained during assembly, and no experimental values
were recomputed to create the repository. Synthetic smoke tests do not establish
bitwise reproduction of the full study. See [VALIDATION](docs/VALIDATION.md) for the
checks actually performed.

## Citation and license

`CITATION.cff` contains the available software metadata. Add the final article and
archived-code DOI after assignment. A software license has not been chosen by the
authors; see [LICENSE_NOTICE](LICENSE_NOTICE.md).
