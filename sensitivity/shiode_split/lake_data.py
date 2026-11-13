"""Read-only Shiode 2024 preparation with chronological 56/14/30 separation.

Important: the original Shiode implementation trained on row-index windows.  The
source record is not forced onto a new complete clock grid here; we preserve its
original timestamps/rows and do not interpolate water temperature.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from settings import (
    METEO_COLS, QC_FLAGS, EXPECTED_DEPTHS, EXPECTED_ROWS,
    EXPECTED_DATA_ONLY_VALID_RATIO, EXPECTED_ORIGINAL90_T0,
)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bool_flags(col: pd.Series) -> np.ndarray:
    x = col.fillna(False).astype(str).str.strip().str.lower()
    allowed = {'false':False, '0':False, '0.0':False, 'true':True, '1':True, '1.0':True}
    bad = ~x.isin(allowed)
    if bad.any():
        raise ValueError(f'Invalid QC values in {col.name}: {x[bad].unique()[:5]}')
    return x.map(allowed).to_numpy(dtype=bool)


def split_bounds(n: int, development=.7, internal_train=.8):
    # Same integer rule used for the matched 56/14/30 sensitivity protocol.
    dev_end = int(n * development)
    train_end = int(dev_end * internal_train)
    if not (0 < train_end < dev_end < n):
        raise ValueError('Need nonempty chronological train, validation and test segments.')
    return train_end, dev_end


def fill_forcing_separately(raw, train_end, dev_end):
    """Keep the original linear weather filling, isolated by the new split."""
    result = np.asarray(raw, dtype=np.float64).copy()
    reports = []
    for name, lo, hi in [('train',0,train_end),('val',train_end,dev_end),('test',dev_end,len(raw))]:
        x = result[lo:hi]
        ix = np.arange(len(x))
        for j, col in enumerate(METEO_COLS):
            valid = np.isfinite(x[:,j])
            if not valid.any():
                raise ValueError(f'{name}: all weather values missing in {col}')
            n_bad = int((~valid).sum())
            if n_bad:
                x[:,j] = np.interp(ix, ix[valid], x[valid,j])
            reports.append(dict(split=name, column=col, filled=n_bad))
    return result, reports


@dataclass
class Prepared:
    time: np.ndarray
    x: np.ndarray
    temp: np.ndarray
    valid: np.ndarray
    depths: np.ndarray
    columns: list
    T0: np.ndarray
    meteo_mean: np.ndarray
    meteo_std: np.ndarray
    train_end: int
    dev_end: int
    source_sha256: str
    report: dict


def prepare(csv_path, *, strict_site=True) -> Prepared:
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Upload the original Shiode 2024 ML input: {path}')
    return prepare_frame(pd.read_csv(path), source_sha256=file_sha256(path), strict_site=strict_site)


def prepare_frame(df, source_sha256='synthetic', strict_site=True):
    df = df.copy()
    required = ['time'] + METEO_COLS
    absent = [c for c in required if c not in df]
    if absent:
        raise ValueError(f'Missing columns: {absent}')

    parsed_time = pd.to_datetime(df['time'], errors='raise')
    if getattr(parsed_time.dt, 'tz', None) is not None or parsed_time.isna().any():
        raise ValueError('time must be timezone-naive and valid.')
    was_sorted = bool(parsed_time.is_monotonic_increasing)
    df['time'] = parsed_time
    # Match the original data_loader_qc.py behavior exactly: sort by time before use.
    df = df.sort_values('time').reset_index(drop=True)
    time = pd.DatetimeIndex(df['time'])
    if time.has_duplicates:
        raise ValueError('Duplicate timestamps are not supported.')
    if strict_site and set(time.year) != {2024}:
        raise ValueError('This package is only for the original Shiode 2024 training record.')

    # Preserve the historical row-index windowing.  Report timestamp gaps but do not
    # insert/drop rows, because that would change the original training conditions.
    if len(time) > 1:
        dt_min = np.diff(time.asi8) / 60_000_000_000
        n_not_10 = int(np.count_nonzero(np.abs(dt_min - 10.0) > 1e-9))
        max_gap = float(np.max(dt_min))
        min_gap = float(np.min(dt_min))
    else:
        n_not_10, max_gap, min_gap = 0, None, None

    candidates = [c for c in df if re.fullmatch(r'\d+(?:\.\d+)?\s*m', str(c).strip())]
    candidates.sort(key=lambda c: float(c.strip().replace(' ', '').replace('m','')))
    depths = np.array([float(c.strip().replace(' ', '').replace('m','')) for c in candidates], dtype=np.float64)
    expected = np.asarray(EXPECTED_DEPTHS, dtype=np.float64)
    if len(depths) != len(expected) or not np.allclose(depths, expected, atol=1e-6):
        raise ValueError(
            'Expected the original Shiode 2024 trained depth columns '
            f'{EXPECTED_DEPTHS}; found {depths.tolist()}. No drop/remap/interpolation is allowed.'
        )
    columns = list(candidates)

    temp = df[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    valid = np.isfinite(temp)
    flag_counts = {}
    for j, c in enumerate(columns):
        for suffix in QC_FLAGS:
            key = f'{c}_{suffix}'
            if key not in df:
                raise ValueError(f'Missing data-only QC flag: {key}; upload *_aligned_qc_wide.csv, not aligned.csv.')
            flag = bool_flags(df[key])
            valid[:,j] &= ~flag
            flag_counts[key] = int(flag.sum())

    n = len(time)
    if strict_site and n != EXPECTED_ROWS:
        raise ValueError(
            f'Expected {EXPECTED_ROWS} rows from the original Shiode 2024 training input; found {n}. '
            'Do not silently use another processed CSV.'
        )

    # Strong provenance fingerprint from the original formal Shiode run.
    original_valid_ratio = float(valid.mean())
    original90_end = int(n * 0.9)
    old_counts = valid[:original90_end].sum(axis=0)
    old_t0 = (
        np.where(valid[:original90_end], temp[:original90_end], 0.).sum(axis=0, dtype=np.float64)
        / np.clip(old_counts, 1, None)
    ).astype(np.float32)
    if strict_site:
        if abs(original_valid_ratio - EXPECTED_DATA_ONLY_VALID_RATIO) > 5e-6:
            raise ValueError(
                f'Shiode provenance mismatch: data-only valid ratio={original_valid_ratio:.9f}, '
                f'expected {EXPECTED_DATA_ONLY_VALID_RATIO:.9f}. Wrong/regenerated CSV is likely.'
            )
        if not np.allclose(old_t0, np.asarray(EXPECTED_ORIGINAL90_T0, dtype=np.float32), atol=2e-4, rtol=0):
            raise ValueError(
                'Shiode provenance mismatch: the original 90% training T0 fingerprint does not match. '
                f'Found {old_t0.tolist()}; expected {EXPECTED_ORIGINAL90_T0}.'
            )

    train_end, dev_end = split_bounds(n)
    raw, filling = fill_forcing_separately(df[METEO_COLS].to_numpy(dtype=np.float64), train_end, dev_end)
    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0)
    std = np.where(std < 1e-6, 1., std)
    norm = (raw - mean) / std
    # 2024 is leap year; preserve the source implementation's 366-day DOY period.
    angle = 2*np.pi * time.dayofyear.to_numpy(dtype=np.float64) / 366.
    x = np.column_stack([norm, np.sin(angle), np.cos(angle)]).astype(np.float32)

    counts = valid[:train_end].sum(axis=0)
    if (counts == 0).any():
        raise ValueError(f'No valid training observation at {np.array(columns)[counts==0].tolist()}')
    T0 = (
        np.where(valid[:train_end], temp[:train_end], 0.).sum(axis=0, dtype=np.float64)
        / counts
    ).astype(np.float32)
    filled = np.where(np.isfinite(temp), temp, 0.).astype(np.float32)

    report = dict(
        n_rows=n,
        train_end_exclusive=train_end,
        development_end_exclusive=dev_end,
        temperature_columns=columns,
        source_depth_mode='original_shiode_2024_11_channels_no_drop_no_remap',
        internal_model_depths_m=depths.tolist(),
        excluded_flags=QC_FLAGS,
        qc_flag_counts=flag_counts,
        data_only_valid_ratio=original_valid_ratio,
        original90_T0_fingerprint=old_t0.tolist(),
        original90_end_exclusive=original90_end,
        input_time_already_sorted=was_sorted,
        timestamp_non10min_steps=n_not_10,
        timestamp_min_step_minutes=min_gap,
        timestamp_max_step_minutes=max_gap,
        forcing_fill=filling,
        split_rule='dev_end=int(N*0.70); train_end=int(dev_end*0.80)',
        split_dependency='meteo normalization and fallback T0 computed from matched-split training rows only',
        temperature_interpolation=False,
        time_reindexing=False,
    )
    report['train'] = dict(start=str(time[0]), end=str(time[train_end-1]), n_rows=train_end)
    report['val'] = dict(start=str(time[train_end]), end=str(time[dev_end-1]), n_rows=dev_end-train_end)
    report['test'] = dict(start=str(time[dev_end]), end=str(time[-1]), n_rows=n-dev_end)

    return Prepared(
        time=time.to_numpy(), x=x, temp=filled, valid=valid, depths=depths.astype(np.float32),
        columns=columns, T0=T0, meteo_mean=mean.astype(np.float32), meteo_std=std.astype(np.float32),
        train_end=train_end, dev_end=dev_end, source_sha256=source_sha256, report=report,
    )


def window_starts(n, train_end, dev_end, cfg, split):
    total = cfg['warmup'] + cfg['target']
    stride = cfg['stride']
    if split == 'train':
        lo, hi = 0, train_end
        starts = np.arange(lo, hi-total+1, stride, dtype=np.int64)
    elif split == 'val':
        # Historical warmup may reach backward into train, while every scored target is in validation.
        target_lo, target_hi = train_end, dev_end
        lo = max(0, target_lo-cfg['warmup'])
        starts = np.arange(lo, target_hi-total+1, stride, dtype=np.int64)
        starts = starts[(starts+cfg['warmup'] >= target_lo) & (starts+total <= target_hi)]
    elif split == 'test':
        # Historical forcing warmup may reach backward into validation; test temperature labels are masked in __getitem__.
        target_lo, target_hi = dev_end, n
        lo = max(0, target_lo-cfg['warmup'])
        starts = np.arange(lo, target_hi-total+1, stride, dtype=np.int64)
        starts = starts[(starts+cfg['warmup'] >= target_lo) & (starts+total <= target_hi)]
    else:
        raise ValueError(split)
    if len(starts) == 0:
        raise ValueError(f'No windows for split={split}; rows={n}, train_end={train_end}, dev_end={dev_end}')
    return starts


class Windows(Dataset):
    def __init__(self, data: Prepared, cfg: dict, split: str):
        self.data,self.cfg,self.split = data,cfg,split
        self.starts = window_starts(len(data.time),data.train_end,data.dev_end,cfg,split)
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s = int(self.starts[i]); e = s+self.cfg['warmup']+self.cfg['target']
        y = self.data.temp[s:e].copy()
        m = self.data.valid[s:e].copy()
        if self.split == 'test':
            # No held-out temperature information is supplied through the warmup labels.
            y[:self.cfg['warmup']] = 0.
            m[:self.cfg['warmup']] = False
        return (torch.from_numpy(self.data.T0.copy()), torch.from_numpy(self.data.x[s:e].copy()),
                torch.from_numpy(y), torch.from_numpy(m.astype(np.float32)), s)


def target_coverage(starts, n, cfg):
    count=np.zeros(n,dtype=np.int32)
    for s in starts:
        a=int(s)+cfg['warmup']; count[a:a+cfg['target']]+=1
    return count
