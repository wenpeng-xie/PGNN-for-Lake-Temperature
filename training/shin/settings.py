"""Frozen Shin protocol. Model definitions/loss weights follow the uploaded snapshot."""
from __future__ import annotations
import copy

VERSION = 'Shin_ML_iso4_retrain_v1'
METEO_COLS = ['solar', 'precip', 'wind', 'air_temp', 'rh', 'pressure']
QC_FLAGS = ['flag_nan', 'flag_range', 'flag_spike', 'flag_stuck']
EXPECTED_DEPTHS = [round(0.25 + 0.2*i, 2) for i in range(18)]
PAPER_DEPTH_MAX = 3.45
COMMON = dict(epochs=200, warmup=1008, target=288, stride=24, hidden=64,
              lr=1e-3, weight_decay=1e-4, grad_clip=5., temp_min=0., temp_max=40.,
              internal_train_ratio=0.8, development_ratio=0.7,
              checkpoint_every=25, num_workers=0, precision='float32')
MODELS = {
 'lstm': dict(batch_size=64, warm_weight=.3, time_weight=0., depth_weight=0.,
              range_weight=0., std_weight=0., gmean_weight=0., heat_weight=0.,
              actual_seeds=[42,1,2,3,4,5,6,7,8,9], export_folder='shin_2025_lstm_base_csv'),
 'rssm': dict(batch_size=256, warm_weight=.2, time_weight=1e-4, depth_weight=1e-4,
              range_weight=.001, std_weight=0., gmean_weight=0., heat_weight=0.,
              update_scale=.05, actual_seeds=[42,1,2,3,4,5,6,7,8,9],
              export_folder='shin_2025_ssm_structure_csv'),
 'pgnn': dict(batch_size=64, warm_weight=.3, time_weight=0., depth_weight=0.,
              range_weight=.001, std_weight=.30, gmean_weight=.20, heat_weight=0.,
              actual_seeds=list(range(10)), export_folder='shin_2025_lstm_structure_sgh_csv'),
}
def configuration(model: str) -> dict:
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    return dict(copy.deepcopy(COMMON), **copy.deepcopy(MODELS[model]), model=model, version=VERSION)
