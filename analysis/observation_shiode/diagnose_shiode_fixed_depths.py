#!/usr/bin/env python3
"""R1-5, step 1A: Shiode observed-profile completeness and event audit.

Read-only analysis of existing observations. NO GLM, ML, training or inference.
Python 3.9+, numpy, pandas; matplotlib for optional PNG figures.
The diagnostic is run by the user, not at package creation time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = '1.0.0'
DEFAULT_ROOT = 'data'
DEPTHS = np.array([25 + 20 * i for i in range(11)], dtype=float) / 100
DATA_FLAGS = ('flag_nan', 'flag_range', 'flag_spike', 'flag_stuck')
PHYSICS_FLAGS = ('flag_vert', 'flag_unstable')
START = '2025-05-08 00:00:00'
END = '2025-11-23 00:00:00'
STEP_MINUTES = 10
WINDOW_STEPS = 18                  # 180 minutes; 19 samples including endpoints
S_START_MIN = 0.50
DROP_ABS_MIN = 0.25
STRUCTURE_MIN_FRACTION = 0.70       # legacy continuous-structure comparator only


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def parse_flag(series: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Unknown flags are NOT silently treated as good. Unexpected literals fail."""
    x = series.astype('string').str.strip().str.lower()
    mapping = {'true': True, 'false': False, '1': True, '0': False,
               '1.0': True, '0.0': False, 't': True, 'f': False}
    v = x.map(mapping)
    unexpected = x.notna() & v.isna()
    require(not bool(unexpected.any()),
            'Unrecognized QC values in %s: %s' % (name, x[unexpected].unique().tolist()))
    known = v.notna().to_numpy(dtype=bool)
    bad = v.fillna(False).to_numpy(dtype=bool)
    return bad, known


def row_std(values: np.ndarray) -> np.ndarray:
    """Legacy event STD: population SD (ddof=0), missing layers skipped."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanstd(values, axis=1, ddof=0)


def legacy_structure(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Historical continuous STD/Gmean rules, for comparison ONLY.

    >=70% finite layers; Gmean linearly fills the nominal depth grid with
    np.interp, including constant edge extension. This does not change inputs.
    The strict analysis below does NOT use interpolation.
    """
    fraction = np.isfinite(values).mean(axis=1)
    sd = row_std(values)
    sd[fraction < STRUCTURE_MIN_FRACTION] = np.nan
    g = np.full(len(values), np.nan)
    for i in np.flatnonzero(fraction >= STRUCTURE_MIN_FRACTION):
        good = np.isfinite(values[i])
        if good.sum() >= 2:
            filled = np.interp(DEPTHS, DEPTHS[good], values[i, good])
            g[i] = float(np.mean(np.abs(np.diff(filled) / np.diff(DEPTHS))))
    return sd, g


def forward_complete(complete_profile: np.ndarray) -> np.ndarray:
    """True only for a FULL 19-sample window on the unchanged regular time grid."""
    n = len(complete_profile)
    out = np.zeros(n, dtype=bool)
    length = WINDOW_STEPS + 1
    if n >= length:
        sums = np.r_[0, np.cumsum(complete_profile.astype(np.int64))]
        out[:n - length + 1] = sums[length:] - sums[:-length] == length
    return out


def candidates(sd: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Port of detect_obs_std_events_abs in Figure(3).ipynb.

    Baseline preserves the original truncated tail and NaN-skipping behavior.
    Strict mode gates candidates BEFORE greedy event-start merging. Thresholds,
    max/min temporal ordering, and 3h start-separation rule are unchanged.
    """
    n = len(sd)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        if eligible is not None and not eligible[i]:
            continue
        if not np.isfinite(sd[i]) or sd[i] < S_START_MIN:
            continue
        j = min(i + WINDOW_STEPS, n - 1)
        w = sd[i:j + 1]
        if np.isfinite(w).sum() < 2:
            continue
        peak = int(np.nanargmax(w))
        trough = int(np.nanargmin(w))
        drop = float(w[peak] - w[trough])
        out[i] = bool(np.isfinite(drop) and drop >= DROP_ABS_MIN and trough > peak)
    return out


def merge_starts(candidate: np.ndarray) -> list[int]:
    """Exact historical greedy rule: next start > previous start + 18 samples."""
    starts: list[int] = []
    last = -10**12
    for x in np.flatnonzero(candidate):
        i = int(x)
        if i > last + WINDOW_STEPS:
            starts.append(i)
            last = i
    return starts


EVENT_COLUMNS = [
    'event_id', 'catalog', 'start_index', 'peak_index', 'min_index', 'end_index',
    'start_time', 'peak_time', 'min_time', 'end_time', 'window_minutes',
    'STD_start_degC', 'STD_peak_degC', 'STD_min_degC', 'drop_degC',
    'peak_to_min_minutes', 'raw_layers_min', 'raw_layers_max',
    'valid_layers_min', 'valid_layers_max', 'raw_depth_set_change_count',
    'valid_depth_set_change_count', 'window_rows_absent', 'nonfinite_temperature_cells',
    'data_quality_rejected_finite_cells', 'QC_flag_unknown_finite_cells',
    'full_3h_window', 'all11_raw_complete', 'all11_data_quality_complete',
    'strict_window_eligible', 'not_eligible_reason',
]


def make_catalog(starts: list[int], sd: np.ndarray, label: str, prefix: str,
                 time: pd.DatetimeIndex, raw_valid: np.ndarray, valid: np.ndarray,
                 present: np.ndarray, unknown_qc: np.ndarray) -> pd.DataFrame:
    rows = []
    for eid, i in enumerate(starts, 1):
        j = min(i + WINDOW_STEPS, len(sd) - 1)
        w = sd[i:j + 1]
        p = i + int(np.nanargmax(w))
        q = i + int(np.nanargmin(w))
        rv, vv = raw_valid[i:j + 1], valid[i:j + 1]
        un = unknown_qc[i:j + 1]
        full = (j - i == WINDOW_STEPS)
        reasons = []
        if not full:
            reasons.append('less_than_full_3h_at_series_end')
        absent_rows = int((~present[i:j + 1]).sum())
        if absent_rows:
            reasons.append('missing_timestamp_rows')
        missing_cells = int((~rv).sum())
        if missing_cells:
            reasons.append('missing_temperature')
        unknown_cells = int((rv & un).sum())
        if unknown_cells:
            reasons.append('unknown_data_quality_flag')
        qc_rejected = int((rv & ~vv & ~un).sum())
        if qc_rejected:
            reasons.append('data_quality_flagged')
        nr, nv = rv.sum(axis=1), vv.sum(axis=1)
        rows.append(dict(
            event_id=f'{prefix}{eid:04d}', catalog=label,
            start_index=i, peak_index=p, min_index=q, end_index=j,
            start_time=str(time[i]), peak_time=str(time[p]), min_time=str(time[q]),
            end_time=str(time[j]), window_minutes=(j-i)*STEP_MINUTES,
            STD_start_degC=float(sd[i]), STD_peak_degC=float(sd[p]),
            STD_min_degC=float(sd[q]), drop_degC=float(sd[p]-sd[q]),
            peak_to_min_minutes=(q-p)*STEP_MINUTES,
            raw_layers_min=int(nr.min()), raw_layers_max=int(nr.max()),
            valid_layers_min=int(nv.min()), valid_layers_max=int(nv.max()),
            raw_depth_set_change_count=int(np.any(rv[1:] != rv[:-1], axis=1).sum()),
            valid_depth_set_change_count=int(np.any(vv[1:] != vv[:-1], axis=1).sum()),
            window_rows_absent=absent_rows,
            nonfinite_temperature_cells=missing_cells,
            data_quality_rejected_finite_cells=qc_rejected,
            QC_flag_unknown_finite_cells=unknown_cells,
            full_3h_window=full, all11_raw_complete=bool(rv.all()),
            all11_data_quality_complete=bool(vv.all()),
            strict_window_eligible=bool(full and vv.all()),
            not_eligible_reason=';'.join(reasons) if reasons else 'eligible',
        ))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def annotate_original_windows(table: pd.DataFrame, strict_events: pd.DataFrame,
                              strict_candidate: np.ndarray,
                              sd_data: np.ndarray) -> pd.DataFrame:
    """Window provenance, NOT an automatic one-to-one physical event matcher."""
    out = table.copy()
    strict_starts = {int(r.start_index): str(r.event_id)
                     for r in strict_events.itertuples(index=False)}
    exact, overlaps, same_peak, states, new_drops = [], [], [], [], []
    for r in out.itertuples(index=False):
        i, j = int(r.start_index), int(r.end_index)
        exact.append(strict_starts.get(i, ''))
        ov = strict_events[(strict_events.start_index <= j) & (strict_events.end_index >= i)]
        overlaps.append(';'.join(ov.event_id.tolist()))
        sm = strict_events[(strict_events.peak_index == r.peak_index) &
                           (strict_events.min_index == r.min_index)]
        same_peak.append(';'.join(sm.event_id.tolist()))
        w = sd_data[i:j + 1]
        new_drops.append(float(np.nanmax(w)-np.nanmin(w)) if np.isfinite(w).sum() >= 2 else np.nan)
        if not r.strict_window_eligible:
            state = 'not_assessable_under_strict_window_rule'
        elif not strict_candidate[i]:
            state = 'does_not_satisfy_event_rule_after_data_QC'
        elif i in strict_starts:
            state = 'retained_same_window_start'
        else:
            state = 'valid_candidate_but_suppressed_by_unchanged_greedy_merge'
        states.append(state)
    out['strict_catalog_same_start_id'] = exact
    out['strict_catalog_same_peak_min_ids'] = same_peak
    out['strict_catalog_overlapping_window_ids_not_equivalence'] = overlaps
    out['status_under_strict_rule'] = states
    out['data_QC_window_max_min_drop_degC'] = new_drops
    return out


def compare_legacy_unified(path: Path, time: pd.DatetimeIndex, raw: np.ndarray) -> dict:
    """Optional check of the old event observation columns, not old model CSVs."""
    if not path.is_file():
        return {'status': 'not_provided', 'path': str(path),
                'note': 'R0 is a raw-column baseline, not independently verified as the published event input.'}
    cols = ['time'] + [f'obs_{z:.2f}m' for z in DEPTHS]
    frame = pd.read_csv(path, usecols=cols)
    t = pd.to_datetime(frame.pop('time'), errors='raise')
    require(t.notna().all() and not t.duplicated().any(), 'Invalid/duplicate times in legacy observation CSV')
    frame.index = pd.DatetimeIndex(t)
    aligned = frame.reindex(time)
    a = aligned.to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(raw)
    max_diff = float(np.max(np.abs(a[both]-raw[both]))) if both.any() else None
    missing_equal = bool(np.array_equal(np.isfinite(a), np.isfinite(raw)))
    agrees = missing_equal and max_diff is not None and max_diff <= 1e-5
    return {'status': 'agrees_within_1e-5_degC' if agrees else 'mismatch_requires_review',
            'path': str(path), 'sha256': sha256(path), 'finite_mask_equal': missing_equal,
            'max_absolute_difference_degC': max_diff,
            'note': 'Only the observation columns were checked; no legacy predictions are used.'}


def make_plots(out: Path, time: pd.DatetimeIndex, cleaned: np.ndarray,
               raw_count: np.ndarray, valid_count: np.ndarray,
               sd_available: np.ndarray, sd_fixed: np.ndarray,
               old_g: np.ndarray, fixed_g: np.ndarray) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    def finish(fig, ax, name):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlim(time[0], time[-1])
        fig.tight_layout()
        fig.savefig(out / name, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.4))
    mesh = ax.pcolormesh(time, DEPTHS, np.ma.masked_invalid(cleaned.T), shading='nearest')
    ax.invert_yaxis()
    ax.set_ylabel('Nominal depth below surface (m)')
    ax.set_title('Shiode 2025 | observations after data-quality screening; gaps are missing/invalid')
    fig.colorbar(mesh, ax=ax, label='Water temperature (degC)')
    finish(fig, ax, '01_observation_heatmap.png')

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.step(time, raw_count, where='post', label='Finite recorded temperatures', alpha=0.75)
    ax.step(time, valid_count, where='post', label='Data-quality-valid depths')
    ax.set_ylim(-0.4, 11.6)
    ax.set_yticks([0, 3, 6, 9, 11])
    ax.set_ylabel('Number of valid depths')
    ax.set_title('Fixed set of 11 nominal depths | no removal solely for instability flags')
    ax.legend(loc='best')
    finish(fig, ax, '02_valid_depth_count.png')

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(time, sd_available, linewidth=0.7, alpha=0.7, label='Available-depth STD, same data-quality mask')
    ax.plot(time, sd_fixed, linewidth=0.8, label='Complete-11-depth STD only')
    ax.set_ylabel('Vertical STD (degC; ddof=0)')
    ax.set_title('Incomplete profiles remain gaps in the fixed-depth series')
    ax.legend(loc='best')
    finish(fig, ax, '03_STD_comparison.png')

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(time, old_g, linewidth=0.7, alpha=0.7, label='Legacy >=70%-depth interpolation comparator')
    ax.plot(time, fixed_g, linewidth=0.8, label='Complete-11-depth Gmean; no filling')
    ax.set_ylabel('Mean absolute temperature gradient (degC/m)')
    ax.set_title('Same nominal depths; legacy interpolation is diagnostic only')
    ax.legend(loc='best')
    finish(fig, ax, '04_Gmean_comparison.png')


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root).expanduser().resolve()
    source = Path(args.obs_csv).expanduser().resolve() if args.obs_csv else root / 'shiode_2025_aligned_qc_wide.csv'
    require(source.is_file(), 'Observation CSV not found: ' + str(source))
    if not args.no_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ImportError('matplotlib is required for PNGs; install it or use --no-figures') from exc
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    require(start.tz is None and end.tz is None and start < end, 'Use increasing, timezone-naive local timestamps')
    dt = pd.Timedelta(minutes=STEP_MINUTES)
    require((end.value-start.value) % dt.value == 0, 'End must lie on the 10-minute grid')
    time = pd.date_range(start, end, freq=dt)
    require(len(time) > WINDOW_STEPS, 'The requested interval must be longer than 3 hours')
    columns = [f'{d:.2f}m' for d in DEPTHS]
    needed = ['time'] + columns + [f'{c}_{f}' for c in columns for f in DATA_FLAGS]
    header = pd.read_csv(source, nrows=0)
    missing = [c for c in needed if c not in header.columns]
    require(not missing, 'Required columns missing (use QC WIDE CSV): ' + ', '.join(missing))
    optional = [f'{c}_{f}' for c in columns for f in PHYSICS_FLAGS + ('qc_bad',) if f'{c}_{f}' in header.columns]
    before = source.stat()
    df = pd.read_csv(source, usecols=needed+optional)
    parsed = pd.to_datetime(df.pop('time'), errors='raise')
    require(parsed.notna().all(), 'Observation time column contains missing timestamps')
    index = pd.DatetimeIndex(parsed)
    require(index.tz is None, 'Timezone-aware input detected; resolve its time convention explicitly first')
    df.index = index
    df = df.loc[(df.index >= start) & (df.index <= end)].sort_index()
    require(len(df) > 0, 'No observations in the requested interval')
    require(not df.index.duplicated().any(), 'Duplicate observation timestamps; no automatic averaging is allowed')
    require(bool(np.all((df.index.asi8-start.value) % dt.value == 0)),
            'Off-grid timestamps found; this script never rounds or silently resamples temperatures')
    present = np.asarray(time.isin(df.index), dtype=bool)
    df = df.reindex(time)  # insert absent rows as missing; NEVER compact time
    raw = np.column_stack([pd.to_numeric(df[c], errors='raise').to_numpy(dtype=float) for c in columns])
    raw_valid = np.isfinite(raw)
    bad = np.zeros_like(raw_valid)
    unknown = np.zeros_like(raw_valid)
    flag_rows = []
    for j, c in enumerate(columns):
        for flag in DATA_FLAGS:
            b, k = parse_flag(df[f'{c}_{flag}'], f'{c}_{flag}')
            bad[:, j] |= b
            unknown[:, j] |= ~k
            flag_rows.append(dict(depth_m=float(DEPTHS[j]), flag=flag, used_to_exclude=True,
                                  flagged_present_rows=int((b & present).sum()),
                                  unknown_present_rows=int((~k & present).sum())))
        for flag in PHYSICS_FLAGS + ('qc_bad',):
            key = f'{c}_{flag}'
            if key in df:
                b, k = parse_flag(df[key], key)
                flag_rows.append(dict(depth_m=float(DEPTHS[j]), flag=flag, used_to_exclude=False,
                                      flagged_present_rows=int((b & present).sum()),
                                      unknown_present_rows=int((~k & present).sum())))
    valid = raw_valid & ~bad & ~unknown
    cleaned = np.where(valid, raw, np.nan)
    full_profile = valid.all(axis=1)
    full_raw = raw_valid.all(axis=1)
    eligible = forward_complete(full_profile)
    n_raw, n_valid = raw_valid.sum(axis=1), valid.sum(axis=1)
    sd_raw = row_std(raw)
    sd_data = row_std(cleaned)
    sd_fixed = np.where(full_profile, sd_data, np.nan)
    legacy_std_raw, legacy_g_raw = legacy_structure(raw)
    legacy_std_data, legacy_g_data = legacy_structure(cleaned)
    g_fixed = np.full(len(time), np.nan)
    g_fixed[full_profile] = np.mean(np.abs(np.diff(cleaned[full_profile], axis=1) / np.diff(DEPTHS)), axis=1)

    # Assertions protect the intended protocol, not a desired event count.
    require(np.allclose(sd_fixed[full_profile], sd_data[full_profile], atol=0, rtol=0), 'STD consistency failure')
    require(np.allclose(g_fixed[full_profile], legacy_g_data[full_profile], atol=1e-12), 'Gradient consistency failure')
    cand_raw = candidates(sd_raw)
    cand_data = candidates(sd_data)
    cand_strict = candidates(sd_fixed, eligible)
    selected_raw, selected_data, selected_strict = map(merge_starts, [cand_raw, cand_data, cand_strict])
    cat_raw = make_catalog(selected_raw, sd_raw, 'R0_raw_available_legacy_logic', 'R', time, raw_valid, valid, present, unknown)
    cat_data = make_catalog(selected_data, sd_data, 'A_data_quality_available_depths', 'A', time, raw_valid, valid, present, unknown)
    cat_strict = make_catalog(selected_strict, sd_fixed, 'B_complete_11_depths_full_3h', 'B', time, raw_valid, valid, present, unknown)
    require(cat_strict.empty or bool(cat_strict.strict_window_eligible.all()), 'Strict catalog contains an ineligible event')
    checks_raw = annotate_original_windows(cat_raw, cat_strict, cand_strict, sd_data)
    checks_data = annotate_original_windows(cat_data, cat_strict, cand_strict, sd_data)

    oldpath = Path(args.legacy_csv).expanduser().resolve() if args.legacy_csv else root / 'runs_qc/unified_plot_csv_all_layers/shiode_unified_sec_layers.csv'
    provenance = compare_legacy_unified(oldpath, time, raw)
    require((before.st_size,before.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns),
            'The input changed during the audit; stop and run again from an unchanged file')
    outroot = Path(args.out_root).expanduser().resolve() if args.out_root else root / 'revision_R1_5'
    out = outroot / ('shiode_step1A_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)

    changed_raw = np.r_[False, np.any(raw_valid[1:] != raw_valid[:-1], axis=1)]
    changed_valid = np.r_[False, np.any(valid[1:] != valid[:-1], axis=1)]
    ts = pd.DataFrame(dict(time=time, row_present=present, n_finite_depths=n_raw,
                           n_data_quality_valid_depths=n_valid,
                           raw_depth_set_changed=changed_raw,
                           valid_depth_set_changed=changed_valid,
                           complete11_raw=full_raw, complete11_data_quality=full_profile,
                           eligible_complete3h_start=eligible,
                           STD_raw_available_legacy_event=sd_raw,
                           STD_data_quality_available_event=sd_data,
                           STD_fixed11=sd_fixed,
                           STD_raw_legacy_structure70=legacy_std_raw,
                           STD_data_quality_legacy_structure70=legacy_std_data,
                           Gmean_raw_legacy_interpolated70=legacy_g_raw,
                           Gmean_data_quality_legacy_interpolated70=legacy_g_data,
                           Gmean_fixed11=g_fixed))
    write_csv(out/'observation_diagnostics.csv', ts)
    write_csv(out/'events_R0_raw_legacy.csv', cat_raw)
    write_csv(out/'events_A_data_quality_available.csv', cat_data)
    write_csv(out/'events_B_fixed11_complete3h.csv', cat_strict)
    write_csv(out/'audit_R0_event_windows.csv', checks_raw)
    write_csv(out/'audit_A_event_windows.csv', checks_data)
    write_csv(out/'QC_flag_summary.csv', pd.DataFrame(flag_rows))
    write_csv(out/'coverage_by_depth.csv', pd.DataFrame(dict(depth_m=DEPTHS,
        finite_cells=raw_valid.sum(axis=0), data_quality_valid_cells=valid.sum(axis=0),
        expected_time_points=len(time), excluded_finite_cells=(raw_valid & ~valid).sum(axis=0))))

    monthly = []
    months = time.to_period('M').astype(str)
    for month in sorted(set(months)):
        m = np.asarray(months == month)
        monthly.append(dict(month=month, expected_time_points=int(m.sum()),
                            source_rows_present=int(present[m].sum()),
                            finite_temperature_cells=int(raw_valid[m].sum()),
                            data_quality_valid_cells=int(valid[m].sum()),
                            complete11_time_points=int(full_profile[m].sum()),
                            complete11_fraction=float(full_profile[m].mean()),
                            eligible_full3h_start_count=int(eligible[m].sum()),
                            fixed11_events_starting_this_month=int(sum(months[i]==month for i in selected_strict))))
    write_csv(out/'coverage_by_month.csv', pd.DataFrame(monthly))
    np.savez_compressed(out/'fixed_depth_observation_reference.npz',
                        time=time.to_numpy(dtype='datetime64[ns]'), depths=DEPTHS,
                        obs_raw=raw, obs_data_quality=cleaned, raw_finite=raw_valid,
                        valid_mask=valid, source_row_present=present,
                        strict_complete_profile=full_profile, strict_window_start_eligible=eligible,
                        strict_event_start_indices=np.asarray(selected_strict, dtype=np.int64),
                        candidate_raw=cand_raw, candidate_data_quality=cand_data,
                        candidate_strict=cand_strict)
    summary = dict(
        version=VERSION, scope='R1-5 step 1A OBSERVATIONS ONLY',
        input_csv=str(source), input_sha256=sha256(source),
        time_start=str(time[0]), time_end=str(time[-1]), timezone='original CSV convention; not converted',
        nominal_depths_m=DEPTHS.tolist(), time_step_minutes=STEP_MINUTES,
        event_window_minutes=STEP_MINUTES*WINDOW_STEPS, samples_per_complete_window=WINDOW_STEPS+1,
        STD_ddof=0, S_start_min_degC=S_START_MIN, drop_min_degC=DROP_ABS_MIN,
        smoothing='none', event_merge='greedy first candidate; next start strictly more than 3h after previous start',
        data_quality_exclusion_flags=list(DATA_FLAGS), unknown_QC='exclude and report',
        instability_flags='not independently used for exclusion',
        strict_temperature_filling='none; no time or depth interpolation',
        legacy_Gmean_comparator='only comparator uses historical >=70% depth interpolation; never used by strict event detection',
        expected_time_points=len(time), absent_time_rows=int((~present).sum()),
        data_quality_complete11_time_points=int(full_profile.sum()),
        strict_eligible_full3h_start_count=int(eligible.sum()),
        event_counts=dict(R0_raw_legacy=len(cat_raw), A_data_quality_available=len(cat_data),
                          B_fixed11_complete3h=len(cat_strict)),
        candidate_counts=dict(R0_raw_legacy=int(cand_raw.sum()), A_data_quality_available=int(cand_data.sum()),
                              B_fixed11_complete3h=int(cand_strict.sum())),
        old_catalog_counts=dict(
            raw_original_windows_strict_eligible=int(cat_raw.strict_window_eligible.sum()),
            raw_original_windows_not_assessable=int((~cat_raw.strict_window_eligible.astype(bool)).sum()),
            quality_available_windows_strict_eligible=int(cat_data.strict_window_eligible.sum()),
            quality_available_windows_not_assessable=int((~cat_data.strict_window_eligible.astype(bool)).sum())),
        legacy_observation_crosscheck=provenance,
        limitations=[
            'Completeness means valid data in the chosen nominal channels, not proof of every sensor being fault-free.',
            'An ineligible event is not assessable under this rule; it is NOT thereby proven artificial.',
            'Strict re-detection applies the eligibility test before merging; counts need not equal simple filtering of the old catalog.',
            'Monthly coverage must be examined before extrapolating results to the full season.',
            'No model outputs were loaded; no model ranking, drop ratios or reviewer-response conclusion has been validated.',
            'Reviewer R1-5 is not closed until model structure/event re-evaluation and final Figure 2 are updated.',
        ])
    write_json(out/'summary.json', summary)
    if not args.no_figures:
        print('Writing diagnostic PNGs ...', flush=True)
        make_plots(out,time,cleaned,n_raw,n_valid,sd_data,sd_fixed,legacy_g_data,g_fixed)
    report = f'''# Shiode R1-5 第1A步：观测诊断（由本地脚本生成）

输入：`{source}`；SHA256：`{summary['input_sha256']}`。
时段：{time[0]}—{time[-1]}；10分钟；固定11个名义深度0.25—2.25 m。

## 运行结果
| 项目 | 数量 |
|---|---:|
| 规则时间轴时刻 | {len(time)} |
| 原文件缺失的时间行 | {int((~present).sum())} |
| 11层均有效的时刻 | {int(full_profile.sum())} |
| 完整3小时候选起点 | {int(eligible.sum())} |
| 原始数值+旧事件算法（R0）事件 | {len(cat_raw)} |
| 数据质控+可用深度（A）事件 | {len(cat_data)} |
| 相同质控+11层完整3小时（B）事件 | {len(cat_strict)} |
| R0原事件中窗口符合严格完整性的数量 | {int(cat_raw.strict_window_eligible.sum())} |

## 先查看这三项
1. `audit_R0_event_windows.csv` / `audit_A_event_windows.csv`：每个原事件为什么保留或不满足完整性。
2. `coverage_by_month.csv`：筛选是否只剩特定月份；不能仅按事件总数宣称稳健。
3. `events_B_fixed11_complete3h.csv`：供下一步模型重评价使用的观测事件表。

## 解释边界
- R0核对结果：`{provenance['status']}`。只有观测列核对一致时，才能把R0称为对原输入的复算。
- A与B使用相同的数据质量筛选；R0到A另含质控差异，不要把两者全归因于固定层数。
- 3小时含起止19个10分钟剖面；不得删除中间缺测后把时间接起来。
- B在候选合并前执行完整性门槛；事件起点变化可能来自原合并规则，不可直接认定新增/消失的物理过程。
- 不完整只能标注无法纳入严格分析，不能直接称为假事件。
- 严格版STD与Gmean都不插值；旧Gmean插值仅出现在复现对照列。
- 本轮没有读取GLM或ML预测，尚未计算四模型STD/Gmean误差或drop ratio。
- 这是观测诊断，不代表R1-5整条已完成；后续需用最终模型预测完成同事件窗口对照并更新图2。
'''
    (out/'REPORT_zh.md').write_text(report, encoding='utf-8')
    (out/'diagnose_shiode_fixed_depths.py').write_bytes(Path(__file__).read_bytes())
    archive = out/'Shiode_R1_5_step1A_return.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)
    print('\nR0 raw legacy events:', len(cat_raw))
    print('A data-quality available-depth events:', len(cat_data))
    print('B fixed11 complete-window events:', len(cat_strict))
    print('Legacy observation cross-check:', provenance['status'])
    print('No models were trained, run or evaluated.')
    print('OUTPUT DIRECTORY:', out)
    print('RETURN FILE:', archive)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=DEFAULT_ROOT, help='Local project run directory')
    ap.add_argument('--obs-csv', help='Optional explicit QC-wide CSV; defaults to ROOT/shiode_2025_aligned_qc_wide.csv')
    ap.add_argument('--legacy-csv', help='Optional old unified CSV, observation columns only; missing file is reported')
    ap.add_argument('--out-root', help='Default ROOT/revision_R1_5; creates a new timestamped folder')
    ap.add_argument('--start', default=START)
    ap.add_argument('--end', default=END)
    ap.add_argument('--no-figures', action='store_true', help='Only tables/NPZ/report; no matplotlib needed')
    args = ap.parse_args()
    try:
        run(args)
        return 0
    except Exception:
        traceback.print_exc()
        print('\nERROR: no valid completion claim. Do not upload a partial output as a completed result.', file=sys.stderr)
        print('Original observations and models were not modified.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
