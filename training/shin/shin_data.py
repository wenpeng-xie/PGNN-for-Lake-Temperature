"""Read-only data preparation with chronological 56/14/30 separation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from settings import METEO_COLS, QC_FLAGS, EXPECTED_DEPTHS


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bool_flags(col: pd.Series) -> np.ndarray:
    # Missing flag means 'not flagged', as in the source. Finiteness is checked separately.
    x = col.fillna(False).astype(str).str.strip().str.lower()
    allowed = {'false':False, '0':False, '0.0':False, 'true':True, '1':True, '1.0':True}
    bad = ~x.isin(allowed)
    if bad.any():
        raise ValueError(f'Invalid QC values in {col.name}: {x[bad].unique()[:5]}')
    return x.map(allowed).to_numpy(dtype=bool)


def split_bounds(n: int, development=.7, internal_train=.8):
    # Match original int(n*0.7) for the external test boundary.
    dev_end = int(n * development)
    train_end = int(dev_end * internal_train)
    if not (0 < train_end < dev_end < n):
        raise ValueError('Need nonempty chronological train, validation and test segments.')
    return train_end, dev_end


def fill_forcing_separately(raw, train_end, dev_end):
    """Original linear weather interpolation, now isolated at both split boundaries.

    This is observed-forcing hindcast preprocessing, not real-time forecast filling.
    No temperature is interpolated and no later segment contributes to an earlier one.
    """
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


def prepare(csv_path, *, strict_shin=True) -> Prepared:
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Upload the original Shin ML input: {path}')
    return prepare_frame(pd.read_csv(path), source_sha256=file_sha256(path), strict_shin=strict_shin)


def prepare_frame(df, source_sha256='synthetic', strict_shin=True):
    df = df.copy()
    required = ['time'] + METEO_COLS
    absent = [c for c in required if c not in df]
    if absent:
        raise ValueError(f'Missing columns: {absent}')
    time = pd.DatetimeIndex(pd.to_datetime(df['time'], errors='raise'))
    if time.tz is not None or time.hasnans or time.has_duplicates or not time.is_monotonic_increasing:
        raise ValueError('time must be timezone-naive, valid, unique and increasing; no silent sorting.')
    if len(time) < 2 or not np.all(np.diff(time.asi8) == 600_000_000_000):
        raise ValueError('Expected a complete 10-minute time grid. Keep missing observations as NaN rows.')
    if strict_shin and set(time.year) != {2025}:
        raise ValueError('This package is only for Shin 2025.')
    candidates = [c for c in df if re.fullmatch(r'\d+(?:\.\d+)?\s*m', str(c).strip())]
    candidates.sort(key=lambda c: float(c.strip().replace(' ', '').replace('m','')))
    if len(candidates) < 2:
        raise ValueError('No usable temperature depth columns.')
    # This reproduces drop_first_temp_layer=True, rather than silently reducing to 17 channels.
    dropped = candidates[0]
    columns = candidates[1:]
    depths = np.array([float(c.replace(' ', '').replace('m','')) for c in columns], dtype=np.float64)
    if np.any(np.diff(depths) <= 0):
        raise ValueError('Depth column names must be strictly increasing and unique.')
    if strict_shin and (len(depths) != 18 or not np.allclose(depths, EXPECTED_DEPTHS, atol=1e-6)):
        raise ValueError(f'Historical Shin trained 18 outputs 0.25..3.65 m. Found {columns}; do not guess.')
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
    train_end, dev_end = split_bounds(n)
    raw, filling = fill_forcing_separately(df[METEO_COLS].to_numpy(dtype=np.float64), train_end, dev_end)
    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0)
    std = np.where(std < 1e-6, 1., std)
    norm = (raw - mean) / std
    angle = 2*np.pi * time.dayofyear.to_numpy(dtype=np.float64) / 366.
    x = np.column_stack([norm, np.sin(angle), np.cos(angle)]).astype(np.float32)
    counts = valid[:train_end].sum(axis=0)
    if (counts == 0).any():
        raise ValueError(f'No valid training observation at {np.array(columns)[counts==0].tolist()}')
    T0 = (np.where(valid[:train_end], temp[:train_end], 0.).sum(axis=0, dtype=np.float64) / counts).astype(np.float32)
    filled = np.where(np.isfinite(temp), temp, 0.).astype(np.float32)
    report = dict(n_rows=n, train_end_exclusive=train_end, development_end_exclusive=dev_end,
                  temperature_columns=columns, dropped_first_column=dropped,
                  nominal_depths_m=depths.tolist(), input_columns=METEO_COLS+['doy_sin','doy_cos'],
                  train_T0=T0.tolist(), meteo_mean_train=mean.tolist(), meteo_std_train=std.tolist(),
                  forcing_fill=filling, qc_flag_counts=flag_counts,
                  excluded_flags=QC_FLAGS, ignored_physics_flags=['flag_vert','flag_unstable','qc_bad'],
                  depth_rule='fixed observation channels; no water-depth interpolation or layer deletion',
                  test_used_for_training_statistics=False, source_sha256=source_sha256)
    for name,lo,hi in [('train',0,train_end),('val',train_end,dev_end),('test',dev_end,n)]:
        report[name] = dict(start=str(time[lo]), end=str(time[hi-1]), n_rows=hi-lo,
                            n_valid_temperature=int(valid[lo:hi].sum()))
    return Prepared(time.to_numpy(dtype='datetime64[ns]'), x, filled, valid, depths, columns, T0,
                    mean, std, train_end, dev_end, source_sha256, report)


def window_starts(n, train_end, dev_end, cfg, split):
    w,t,s = cfg['warmup'],cfg['target'],cfg['stride']
    if min(w,t,s) <= 0:
        raise ValueError('Positive warmup/target/stride required.')
    if split == 'train':
        starts = np.arange(0, train_end-w-t+1, s, dtype=np.int64)
        target_lo, target_hi = w, train_end
    else:
        lo,hi = (train_end,dev_end) if split=='val' else (dev_end,n) if split=='test' else (None,None)
        if lo is None or lo<w or hi-lo<t:
            raise ValueError(f'{split}: insufficient historical weather or target length.')
        first,last = lo-w, hi-w-t
        starts = np.arange(first, last+1, s, dtype=np.int64)
        if starts.size and starts[-1] != last:
            starts = np.r_[starts, np.int64(last)]
        target_lo,target_hi = lo,hi
    if not len(starts):
        raise ValueError(f'No {split} windows; need a longer record.')
    assert np.all(starts+w >= target_lo)
    assert np.all(starts+w+t <= target_hi)
    assert np.all(starts>=0)
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
        if self.split != 'train':
            # Only historical meteorology is allowed in the warmup, never its temperature labels.
            y[:self.cfg['warmup']] = 0.
            m[:self.cfg['warmup']] = False
        return (torch.from_numpy(self.data.T0.copy()), torch.from_numpy(self.data.x[s:e].copy()),
                torch.from_numpy(y), torch.from_numpy(m.astype(np.float32)), s)


def target_coverage(starts, n, cfg):
    count=np.zeros(n,dtype=np.int32)
    for s in starts:
        a=int(s)+cfg['warmup']; count[a:a+cfg['target']]+=1
    return count
