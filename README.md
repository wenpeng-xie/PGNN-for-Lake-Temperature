# Shallow-lake temperature models and hierarchical diagnostics

Code accompanying **High-Frequency Observations Reveal Trade-Offs Between Profile
Accuracy and Thermal-Structure Fidelity in Small Shallow Lake Temperature Models**.

This repository contains the  source versions identified for the JoHRS revision:
LSTM, residual state-space model (rSSM), and partially physics-guided neural network
(PGNN); thermal-structure and event diagnostics; sensitivity
analyses; and publication-figure workflows.

## Installation

Use a Python environment with NumPy, pandas, Matplotlib, PyTorch and tqdm:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py list
```


