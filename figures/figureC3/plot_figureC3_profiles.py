#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

YAMA_DEPTHS = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], dtype=float)
SHIODE_DEPTHS = np.array([0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85, 2.05, 2.25], dtype=float)

MODEL_ORDER = ['GLM', 'LSTM', 'rSSM', 'PGNN']
COLORS = {
    'Observed': '#000000',
    'GLM': '#666666',
    'LSTM': '#1f77b4',
    'rSSM': '#ff7f0e',
    'PGNN': '#2ca02c',
}

YAMA_MEMBER_POLICY = {
    'LSTM': ['base'] + [f'seed_{i}' for i in range(1, 10)],
    'rSSM': ['base'] + [f'seed_{i}' for i in range(1, 10)],
    'PGNN': ['seed3', 'seed4', 'seed5', 'seed6', 'seed7', 'seed9', 'seed14', 'seed15', 'seed16', 'seed20'],
}
SHIODE_MEMBER_POLICY = {
    'LSTM': ['base'] + [f'seed{i}' for i in range(1, 10)],
    'rSSM': ['base'] + [f'seed{i}' for i in range(1, 10)],
    'PGNN': ['seed0', 'seed1', 'seed3', 'seed5', 'seed6', 'seed7', 'seed9', 'seed41', 'seed100', 'seed256'],
}


@dataclass(frozen=True)
class RepresentativeEvent:
    site: str
    event_id: str
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    min_time: pd.Timestamp
    end_time: pd.Timestamp
    drop_degC: float
    std_peak_degC: float
    std_min_degC: float


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def find_file(root: Path, name: str) -> Path:
    p = root / name
    if p.is_file():
        return p
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f'Could not find {name} under {root}')
    return hits[0]


def nearest_depth_indices(available: Sequence[float], targets: Sequence[float], tol: float = 0.12) -> List[int]:
    a = np.asarray(available, dtype=float)
    out: List[int] = []
    for target in targets:
        d = np.abs(a - float(target))
        i = int(np.argmin(d))
        if d[i] > tol:
            raise ValueError(f'Cannot match target depth {target}; available={a.tolist()}')
        out.append(i)
    if len(set(out)) != len(out):
        raise ValueError(f'Duplicate depth mapping: targets={targets}, indices={out}, available={a.tolist()}')
    return out


def exact_time_index(times: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    hits = np.flatnonzero(times.to_numpy(dtype='datetime64[ns]') == np.datetime64(target, 'ns'))
    if len(hits) != 1:
        raise ValueError(f'Could not uniquely locate time {target}; hits={len(hits)}')
    return int(hits[0])


def nearest_grid_time(times: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    ns = times.to_numpy(dtype='datetime64[ns]').astype('int64')
    i = int(np.argmin(np.abs(ns - np.datetime64(target, 'ns').astype('int64'))))
    return pd.Timestamp(times[i])


def representative_times(event: RepresentativeEvent, available_times: pd.DatetimeIndex) -> Dict[str, pd.Timestamp]:
    midpoint = event.peak_time + (event.min_time - event.peak_time) / 2
    return {
        'Before': nearest_grid_time(available_times, event.peak_time),
        'During': nearest_grid_time(available_times, midpoint),
        'After': nearest_grid_time(available_times, event.min_time),
    }


def load_top_yamanokami_event(event_zip: Path) -> Tuple[RepresentativeEvent, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    with zipfile.ZipFile(event_zip) as zf:
        events = pd.read_csv(io.BytesIO(zf.read('yamanokami/core10/events_B.csv')))
        obs_npz = np.load(io.BytesIO(zf.read('yamanokami/core10/observations_and_masks.npz')))
    for c in ['start_time', 'peak_time', 'min_time', 'end_time']:
        events[c] = pd.to_datetime(events[c])
    row = events.sort_values(['drop_degC', 'start_time'], ascending=[False, True]).iloc[0]
    event = RepresentativeEvent(
        site='Yamanokami', event_id=str(row['event_id']),
        start_time=pd.Timestamp(row['start_time']), peak_time=pd.Timestamp(row['peak_time']),
        min_time=pd.Timestamp(row['min_time']), end_time=pd.Timestamp(row['end_time']),
        drop_degC=float(row['drop_degC']), std_peak_degC=float(row['STD_peak_degC']),
        std_min_degC=float(row['STD_min_degC'])
    )
    time = pd.to_datetime(obs_npz['time_ns'].astype('datetime64[ns]'))
    depths = np.asarray(obs_npz['depths'], dtype=float)
    temp = np.asarray(obs_npz['temperature_raw'], dtype=float)
    valid = np.asarray(obs_npz['valid_data_only'], dtype=bool)
    require(np.allclose(depths, YAMA_DEPTHS), f'Unexpected Yamanokami depths: {depths}')
    temp = np.where(valid, temp, np.nan)
    return event, pd.DatetimeIndex(time), depths, temp


def load_top_shiode_event(event_zip: Path) -> RepresentativeEvent:
    with zipfile.ZipFile(event_zip) as zf:
        events = pd.read_csv(io.BytesIO(zf.read('input_events_B_fixed11_complete3h.csv')))
    for c in ['start_time', 'peak_time', 'min_time', 'end_time']:
        events[c] = pd.to_datetime(events[c])
    row = events.sort_values(['drop_degC', 'start_time'], ascending=[False, True]).iloc[0]
    return RepresentativeEvent(
        site='Shiode', event_id=str(row['event_id']),
        start_time=pd.Timestamp(row['start_time']), peak_time=pd.Timestamp(row['peak_time']),
        min_time=pd.Timestamp(row['min_time']), end_time=pd.Timestamp(row['end_time']),
        drop_degC=float(row['drop_degC']), std_peak_degC=float(row['STD_peak_degC']),
        std_min_degC=float(row['STD_min_degC'])
    )


def read_npz_from_zip(zf: zipfile.ZipFile, path: str) -> dict:
    with zf.open(path) as f:
        npz = np.load(io.BytesIO(f.read()))
        return {k: npz[k] for k in npz.files}


def read_npz_from_tar(tf: tarfile.TarFile, path: str) -> dict:
    f = tf.extractfile(path)
    if f is None:
        raise FileNotFoundError(path)
    npz = np.load(io.BytesIO(f.read()))
    return {k: npz[k] for k in npz.files}


def extract_npz_profile(data: dict, target_time: pd.Timestamp, target_depths: Sequence[float]) -> np.ndarray:
    time = pd.to_datetime(np.asarray(data['time']).astype('datetime64[ns]'))
    i = exact_time_index(pd.DatetimeIndex(time), target_time)
    depth_key = 'matched_obs_depths' if 'matched_obs_depths' in data else 'depths'
    depths = np.asarray(data[depth_key], dtype=float)
    temp_key = 'pred_sel' if 'pred_sel' in data else 'pred'
    temp = np.asarray(data[temp_key], dtype=float)
    idxs = nearest_depth_indices(depths, target_depths)
    return temp[i, idxs].astype(float)


def extract_npz_observation_profile(data: dict, target_time: pd.Timestamp, target_depths: Sequence[float]) -> np.ndarray:
    time = pd.to_datetime(np.asarray(data['time']).astype('datetime64[ns]'))
    i = exact_time_index(pd.DatetimeIndex(time), target_time)
    depths = np.asarray(data['matched_obs_depths'], dtype=float)
    obs = np.asarray(data['obs_sel'], dtype=float)
    mask = np.asarray(data['mask_sel'], dtype=float) > 0.5
    idxs = nearest_depth_indices(depths, target_depths)
    profile = obs[i, idxs].astype(float)
    profile[~mask[i, idxs]] = np.nan
    return profile


def extract_csv_profile(zf: zipfile.ZipFile, path: str, target_time: pd.Timestamp, target_depths: Sequence[float]) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(zf.read(path)))
    df['time'] = pd.to_datetime(df['time'])
    ixs = np.flatnonzero(df['time'].to_numpy(dtype='datetime64[ns]') == np.datetime64(target_time, 'ns'))
    if len(ixs) != 1:
        raise ValueError(f'{path}: cannot uniquely find {target_time}; hits={len(ixs)}')
    depth_cols = [c for c in df.columns if c.endswith('m')]
    depths = [float(c[:-1]) for c in depth_cols]
    idxs = nearest_depth_indices(depths, target_depths)
    return df.iloc[int(ixs[0])][[depth_cols[i] for i in idxs]].to_numpy(dtype=float)


def load_yama_glm_profiles(glm_tar: Path, times: Dict[str, pd.Timestamp]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with tarfile.open(glm_tar, 'r:gz') as tf:
        names = sorted([n for n in tf.getnames() if '/predictions/' in n and n.endswith('.npz')])
        require(len(names) == 10, f'Expected 10 Yamanokami GLM members, found {len(names)}')
        for name in names:
            member = Path(name).stem
            d = read_npz_from_tar(tf, name)
            out[member] = {stage: extract_npz_profile(d, t, YAMA_DEPTHS) for stage, t in times.items()}
    return out


def load_shiode_glm_profiles(glm_tar: Path, times: Dict[str, pd.Timestamp]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with tarfile.open(glm_tar, 'r:gz') as tf:
        names = sorted([n for n in tf.getnames() if n.startswith('predictions/') and n.endswith('.npz')])
        require(len(names) == 10, f'Expected 10 Shiode GLM members, found {len(names)}')
        for name in names:
            member = Path(name).stem
            d = read_npz_from_tar(tf, name)
            out[member] = {stage: extract_npz_profile(d, t, SHIODE_DEPTHS) for stage, t in times.items()}
    return out


def load_yama_ml_profiles(ml_zip: Path, pgnn_added_zip: Path, times: Dict[str, pd.Timestamp]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {m: {} for m in ['LSTM', 'rSSM', 'PGNN']}
    with zipfile.ZipFile(ml_zip) as zf:
        paths = {
            'LSTM': {'base': 'evidence/runs_qc/yamanokami_2024_lstm_overlap_eval_base/lstm_overlap_avg_predictions.npz'},
            'rSSM': {'base': 'evidence/runs_qc/yamanokami_2024_ssm_overlap_eval_base/ssm_overlap_avg_predictions.npz'},
            'PGNN': {},
        }
        for i in range(1, 10):
            paths['LSTM'][f'seed_{i}'] = f'evidence/runs_qc/yamanokami_2024_lstm_overlap_eval_base/seed_{i}/lstm_overlap_avg_predictions.npz'
            paths['rSSM'][f'seed_{i}'] = f'evidence/runs_qc/yamanokami_2024_ssm_overlap_eval_structure/seed_{i}/ssm_overlap_avg_predictions.npz'
        for seed in [3, 4, 5, 6, 7, 9]:
            paths['PGNN'][f'seed{seed}'] = f'evidence/runs_qc/yamanokami_2024_lstm_overlap_eval_structure_sgh/seed{seed}/lstm_overlap_avg_predictions.npz'
        for model, members in paths.items():
            for member, path in members.items():
                d = read_npz_from_zip(zf, path)
                out[model][member] = {stage: extract_npz_profile(d, t, YAMA_DEPTHS) for stage, t in times.items()}
    with zipfile.ZipFile(pgnn_added_zip) as zf:
        normalized = {n.replace('\\', '/'): n for n in zf.namelist()}
        for seed in [14, 15, 16, 20]:
            key = f'seed{seed}/lstm_overlap_avg_predictions.npz'
            require(key in normalized, f'Missing {key}')
            d = read_npz_from_zip(zf, normalized[key])
            out['PGNN'][f'seed{seed}'] = {stage: extract_npz_profile(d, t, YAMA_DEPTHS) for stage, t in times.items()}
    for model, wanted in YAMA_MEMBER_POLICY.items():
        require(set(out[model]) == set(wanted), f'Yamanokami {model} member mismatch: got={sorted(out[model])}, wanted={wanted}')
        out[model] = {m: out[model][m] for m in wanted}
    return out


def load_shiode_ml_profiles(ml_zip: Path, times: Dict[str, pd.Timestamp]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {m: {} for m in ['LSTM', 'rSSM', 'PGNN']}
    with zipfile.ZipFile(ml_zip) as zf:
        base_lstm_path = 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_base/lstm_overlap_avg_predictions.npz'
        paths = {
            'LSTM': {'base': base_lstm_path},
            'rSSM': {'base': 'evidence/runs_qc/shiode_2025_ssm_overlap_eval_base/ssm_overlap_avg_predictions.npz'},
            'PGNN': {
                'seed0': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed0/lstm_overlap_avg_predictions.npz',
                'seed1': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed1/lstm_overlap_avg_predictions.npz',
                'seed3': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed3/lstm_overlap_avg_predictions.npz',
                'seed5': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed5/lstm_overlap_avg_predictions.npz',
                'seed6': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed6/lstm_overlap_avg_predictions.npz',
                'seed7': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed7/lstm_overlap_avg_predictions.npz',
                'seed9': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed9/lstm_overlap_avg_predictions.npz',
                'seed256': 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh_1/seed256/lstm_overlap_avg_predictions.npz',
            },
        }
        for i in range(1, 10):
            paths['LSTM'][f'seed{i}'] = f'evidence/runs_qc/shiode_2025_lstm_overlap_eval_base/seed_{i}/lstm_overlap_avg_predictions.npz'
            paths['rSSM'][f'seed{i}'] = f'evidence/runs_qc/shiode_2025_ssm_overlap_eval_structure/seed_{i}/ssm_overlap_avg_predictions.npz'
        for model, members in paths.items():
            for member, path in members.items():
                d = read_npz_from_zip(zf, path)
                out[model][member] = {stage: extract_npz_profile(d, t, SHIODE_DEPTHS) for stage, t in times.items()}
        for seed in [41, 100]:
            member = f'seed{seed}'
            path = f'evidence/runs_qc/shiode_2025_lstm_structure_sgh_csv/shiode_lstm_structure_sgh_seed{seed}.csv'
            out['PGNN'][member] = {stage: extract_csv_profile(zf, path, t, SHIODE_DEPTHS) for stage, t in times.items()}
    for model, wanted in SHIODE_MEMBER_POLICY.items():
        require(set(out[model]) == set(wanted), f'Shiode {model} member mismatch: got={sorted(out[model])}, wanted={wanted}')
        out[model] = {m: out[model][m] for m in wanted}
    return out



def load_shiode_observations(step1a_zip: Path, times: Dict[str, pd.Timestamp]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with zipfile.ZipFile(step1a_zip) as zf:
        npz = np.load(io.BytesIO(zf.read('fixed_depth_observation_reference.npz')))
        time = pd.to_datetime(np.asarray(npz['time']).astype('datetime64[ns]'))
        depths = np.asarray(npz['depths'], dtype=float)
        obs = np.asarray(npz['obs_data_quality'], dtype=float)
        valid = np.asarray(npz['valid_mask'], dtype=bool)
    require(np.allclose(depths, SHIODE_DEPTHS), f'Unexpected Shiode observation depths: {depths}')
    out: Dict[str, np.ndarray] = {}
    time_index = pd.DatetimeIndex(time)
    for stage, t in times.items():
        i = exact_time_index(time_index, t)
        profile = obs[i, :].astype(float)
        profile[~valid[i, :]] = np.nan
        require(np.isfinite(profile).all(), f'Shiode {stage} profile is not complete at {t}')
        out[stage] = profile
    return depths, out

def summarize_members(member_profiles: Dict[str, Dict[str, np.ndarray]], stages: Sequence[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for stage in stages:
        stack = np.stack([member_profiles[m][stage] for m in member_profiles], axis=0).astype(float)
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1)
        out[stage] = (mean, sd, stack.shape[0])
    return out


def build_long_table(
    site: str,
    event: RepresentativeEvent,
    depths: np.ndarray,
    stage_times: Dict[str, pd.Timestamp],
    observed: Dict[str, np.ndarray],
    model_profiles: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> pd.DataFrame:
    rows = []
    for stage, time in stage_times.items():
        for depth, value in zip(depths, observed[stage]):
            rows.append(dict(site=site, event_id=event.event_id, stage=stage, time=time, depth_m=depth,
                             series='Observed', mean_temperature_degC=value, sd_temperature_degC=0.0, n_members=1))
        for model in MODEL_ORDER:
            stack = np.stack([model_profiles[model][m][stage] for m in model_profiles[model]], axis=0)
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1)
            for depth, mu, sigma in zip(depths, mean, sd):
                rows.append(dict(site=site, event_id=event.event_id, stage=stage, time=time, depth_m=depth,
                                 series=model, mean_temperature_degC=mu, sd_temperature_degC=sigma,
                                 n_members=stack.shape[0]))
    return pd.DataFrame(rows)


def plot_profiles(
    outdir: Path,
    site_payloads: Dict[str, dict],
    dpi: int,
) -> None:
    stages = ['Before', 'During', 'After']
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.8), sharey=False)
    letters = list('abcdef')

    for row, site in enumerate(['Yamanokami', 'Shiode']):
        payload = site_payloads[site]
        depths = payload['depths']
        observed = payload['observed']
        models = payload['models']
        event = payload['event']
        times = payload['times']

        # row-specific common x range
        vals = []
        for stage in stages:
            vals.extend(observed[stage][np.isfinite(observed[stage])].tolist())
            for model in MODEL_ORDER:
                stack = np.stack([models[model][m][stage] for m in models[model]], axis=0)
                vals.extend(stack[np.isfinite(stack)].tolist())
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = max(0.4, 0.08 * (hi - lo))

        for col, stage in enumerate(stages):
            ax = axes[row, col]
            ax.plot(observed[stage], depths, color=COLORS['Observed'], marker='o', markersize=3.5,
                    linewidth=2.2, label='Observed', zorder=6)
            for model in MODEL_ORDER:
                stack = np.stack([models[model][m][stage] for m in models[model]], axis=0)
                mean = np.nanmean(stack, axis=0)
                sd = np.nanstd(stack, axis=0, ddof=1)
                ax.fill_betweenx(depths, mean - sd, mean + sd, color=COLORS[model], alpha=0.16, linewidth=0)
                ax.plot(mean, depths, color=COLORS[model], linewidth=1.7, label=model, zorder=4)
            ax.invert_yaxis()
            ax.set_xlim(lo - pad, hi + pad)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_xlabel('Water temperature (°C)')
            if col == 0:
                ax.set_ylabel('Depth (m)')
            subtitle = {
                'Before': 'Before: observed STD peak',
                'During': 'During: midpoint of peak-to-minimum interval',
                'After': 'After: observed STD minimum',
            }[stage]
            ax.set_title(f'({letters[row*3+col]}) {site} — {stage}\n{times[stage]:%Y-%m-%d %H:%M}\n{subtitle}', fontsize=9.5)

        axes[row, 1].text(
            0.5, -0.27,
            f"Representative event {event.event_id}: observed STD drop = {event.drop_degC:.3f} °C",
            transform=axes[row, 1].transAxes, ha='center', va='top', fontsize=9
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l)); seen.add(l)
    fig.legend([u[0] for u in uniq], [u[1] for u in uniq], loc='upper center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        'Figure C3. Representative vertical temperature profiles before, during, and after rapid thermal-structure weakening',
        fontsize=12.5, y=1.02
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94], h_pad=3.2, w_pad=1.6)

    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / 'Figure_C3_representative_profiles'
    fig.savefig(base.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--outdir', type=Path, default=None)
    p.add_argument('--dpi', type=int, default=500)
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    outdir = (args.outdir.expanduser().resolve() if args.outdir else root / 'revision_figures' / 'FigureC3_profiles_v1')

    yama_events_zip = find_file(root, 'Shin_Yamanokami_R1_5_events_return.zip')
    shiode_events_zip = find_file(root, 'Shiode_R1_5_step1B_return.zip')
    shiode_step1a_zip = find_file(root, 'Shiode_R1_5_step1A_return.zip')
    yama_glm = find_file(root, 'GLM_yamanokami_return.tar.gz')
    shiode_glm = find_file(root, 'GLM_shiode_test_return.tar.gz')
    yama_ml = find_file(root, 'Yamanokami_ML_original_batch_20261024_170442_830097.zip')
    yama_pgnn_added = find_file(root, 'Yamanokami_PGNN_added_seeds_20261024_173009.zip')
    shiode_ml = find_file(root, 'Shiode_ML_final_members_20261025_181111_032810.zip')

    y_event, y_time, y_depths, y_obs_matrix = load_top_yamanokami_event(yama_events_zip)
    y_times = representative_times(y_event, y_time)
    y_observed = {stage: y_obs_matrix[exact_time_index(y_time, t), :].copy() for stage, t in y_times.items()}
    y_models: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    y_models['GLM'] = load_yama_glm_profiles(yama_glm, y_times)
    y_models.update(load_yama_ml_profiles(yama_ml, yama_pgnn_added, y_times))

    s_event = load_top_shiode_event(shiode_events_zip)
    # use the base LSTM archive as the observed profile source on the exact common time grid
    with zipfile.ZipFile(shiode_ml) as zf:
        base = read_npz_from_zip(zf, 'evidence/runs_qc/shiode_2025_lstm_overlap_eval_base/lstm_overlap_avg_predictions.npz')
        s_available_time = pd.to_datetime(np.asarray(base['time']).astype('datetime64[ns]'))
    s_times = representative_times(s_event, pd.DatetimeIndex(s_available_time))
    s_depths, s_observed = load_shiode_observations(shiode_step1a_zip, s_times)
    s_ml_models = load_shiode_ml_profiles(shiode_ml, s_times)
    s_models: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {'GLM': load_shiode_glm_profiles(shiode_glm, s_times)}
    s_models.update(s_ml_models)

    site_payloads = {
        'Yamanokami': dict(event=y_event, depths=y_depths, times=y_times, observed=y_observed, models=y_models),
        'Shiode': dict(event=s_event, depths=s_depths, times=s_times, observed=s_observed, models=s_models),
    }

    tables = [
        build_long_table('Yamanokami', y_event, y_depths, y_times, y_observed, y_models),
        build_long_table('Shiode', s_event, s_depths, s_times, s_observed, s_models),
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    profile_table = pd.concat(tables, ignore_index=True)
    profile_table.to_csv(outdir / 'Figure_C3_profile_values.csv', index=False, encoding='utf-8-sig')

    event_rows = []
    for event, times in [(y_event, y_times), (s_event, s_times)]:
        event_rows.append({
            'site': event.site, 'event_id': event.event_id, 'start_time': event.start_time,
            'peak_time': event.peak_time, 'during_time': times['During'], 'min_time': event.min_time,
            'end_time': event.end_time, 'observed_drop_degC': event.drop_degC,
            'observed_STD_peak_degC': event.std_peak_degC, 'observed_STD_min_degC': event.std_min_degC,
            'selection_rule': 'largest observed drop_degC within the final fixed-depth complete-window event catalog',
        })
    pd.DataFrame(event_rows).to_csv(outdir / 'Figure_C3_selected_events_and_times.csv', index=False, encoding='utf-8-sig')

    provenance = {
        'figure': 'Supplementary Figure C3',
        'event_selection': 'One representative event per site: the largest observed drop_degC in the final event catalog. No model result used in selection.',
        'profile_times': {
            'Before': 'observed STD peak time',
            'During': 'nearest 10-min timestamp to the midpoint between observed peak and subsequent minimum',
            'After': 'observed STD minimum time',
        },
        'aggregation': 'For each model and stage, profile mean and sample SD across the 10 frozen members.',
        'fixed_depths_m': {'Yamanokami': YAMA_DEPTHS.tolist(), 'Shiode': SHIODE_DEPTHS.tolist()},
        'inputs': {
            'Yamanokami events/observations': str(yama_events_zip),
            'Shiode events': str(shiode_events_zip),
            'Shiode observations': str(shiode_step1a_zip),
            'Yamanokami GLM': str(yama_glm),
            'Shiode GLM': str(shiode_glm),
            'Yamanokami ML': str(yama_ml),
            'Yamanokami added PGNN': str(yama_pgnn_added),
            'Shiode ML and observations': str(shiode_ml),
        },
        'member_policies': {'Yamanokami': YAMA_MEMBER_POLICY, 'Shiode': SHIODE_MEMBER_POLICY},
    }
    (outdir / 'Figure_C3_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')

    plot_profiles(outdir, site_payloads, args.dpi)

    print('Saved outputs to:', outdir)
    for row in event_rows:
        print(row)


if __name__ == '__main__':
    main()
