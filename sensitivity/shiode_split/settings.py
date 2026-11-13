"""Frozen Shiode matched-split sensitivity protocol.

This package reproduces the final Shiode 2024 ML training geometry and model
hyperparameters, changing only the cross-year evaluation strategy to a
within-year chronological 56/14/30 sensitivity split for Reviewer 2-5.
"""
from __future__ import annotations
import copy

VERSION = 'Shiode_ML_matchedsplit_iso4_v1_0'
METEO_COLS = ['solar', 'precip', 'wind', 'air_temp', 'rh', 'pressure']
QC_FLAGS = ['flag_nan', 'flag_range', 'flag_spike', 'flag_stuck']
EXPECTED_ROWS = 33907
EXPECTED_DEPTHS = [round(0.20 + 0.20*i, 2) for i in range(11)]
# Provenance fingerprint from the original Shiode 2024 training run.
EXPECTED_DATA_ONLY_VALID_RATIO = 0.983138382434845
EXPECTED_ORIGINAL90_T0 = [
    20.866962, 20.409883, 20.31605, 20.164125, 19.962454,
    19.753063, 19.671824, 19.675581, 19.467636, 19.19591, 18.960625,
]
ANALYSIS_DEPTH_MAX = 2.20

COMMON = dict(
    epochs=200, warmup=1008, target=288, stride=24, hidden=64,
    lr=1e-3, weight_decay=1e-4, grad_clip=5.0,
    temp_min=0.0, temp_max=40.0,
    internal_train_ratio=0.8, development_ratio=0.7,
    checkpoint_every=25, num_workers=0, precision='float32',
)

MODELS = {
    'lstm': dict(
        batch_size=64, warm_weight=0.3,
        time_weight=0.0, depth_weight=0.0, range_weight=0.0,
        std_weight=0.0, gmean_weight=0.0, heat_weight=0.0,
        actual_seeds=[42,1,2,3,4,5,6,7,8,9],
        export_folder='shiode_2024_matchedsplit_lstm_csv',
    ),
    'rssm': dict(
        batch_size=256, warm_weight=0.2,
        time_weight=1e-4, depth_weight=1e-4, range_weight=0.001,
        std_weight=0.0, gmean_weight=0.0, heat_weight=0.0,
        update_scale=0.05,
        actual_seeds=[42,1,2,3,4,5,6,7,8,9],
        export_folder='shiode_2024_matchedsplit_rssm_csv',
    ),
    'pgnn': dict(
        batch_size=64, warm_weight=0.3,
        time_weight=0.0, depth_weight=0.0, range_weight=0.001,
        std_weight=0.30, gmean_weight=0.20, heat_weight=0.0,
        actual_seeds=list(range(10)),
        export_folder='shiode_2024_matchedsplit_pgnn_csv',
    ),
}

def configuration(model: str) -> dict:
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    return dict(copy.deepcopy(COMMON), **copy.deepcopy(MODELS[model]),
                model=model, version=VERSION)
