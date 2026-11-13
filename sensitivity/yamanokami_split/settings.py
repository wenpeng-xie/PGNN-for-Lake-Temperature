"""Frozen Yamanokami matched-split sensitivity protocol.

v1.4 restores the actual Yamanokami 2025 ML training geometry verified from the
original prediction archive: 38,449 time steps and 14 trained temperature
channels after dropping the shallowest 0.05-m source layer.
"""
from __future__ import annotations
import copy

VERSION = 'Yamanokami_ML_matchedsplit_iso4_v1_4'
METEO_COLS = ['solar', 'precip', 'wind', 'air_temp', 'rh', 'pressure']
QC_FLAGS = ['flag_nan', 'flag_range', 'flag_spike', 'flag_stuck']
EXPECTED_ROWS = 38449
EXPECTED_START = '2025-03-01 00:00:00'
EXPECTED_END = '2025-11-23 00:00:00'
EXPECTED_RAW_DEPTHS = [round(0.05 + 0.20*i, 2) for i in range(15)]
EXPECTED_DEPTHS = EXPECTED_RAW_DEPTHS[1:]  # original drop_first_temp_layer=True -> 14 channels
ANALYSIS_DEPTH_MAX = 2.85

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
        export_folder='yamanokami_2025_matchedsplit_lstm_csv',
    ),
    'rssm': dict(
        batch_size=256, warm_weight=0.2,
        time_weight=1e-4, depth_weight=1e-4, range_weight=0.001,
        std_weight=0.0, gmean_weight=0.0, heat_weight=0.0,
        update_scale=0.05,
        actual_seeds=[42,1,2,3,4,5,6,7,8,9],
        export_folder='yamanokami_2025_matchedsplit_rssm_csv',
    ),
    'pgnn': dict(
        batch_size=64, warm_weight=0.3,
        time_weight=0.0, depth_weight=0.0, range_weight=0.001,
        std_weight=0.30, gmean_weight=0.20, heat_weight=0.0,
        actual_seeds=list(range(10)),
        export_folder='yamanokami_2025_matchedsplit_pgnn_csv',
    ),
}

def configuration(model: str) -> dict:
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    return dict(copy.deepcopy(COMMON), **copy.deepcopy(MODELS[model]),
                model=model, version=VERSION)
