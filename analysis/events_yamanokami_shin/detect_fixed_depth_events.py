#!/usr/bin/env python3
"""Read an existing precheck ZIP and audit observed STD-drop events at two sites.
No model output, training, inference, imputation, or threshold tuning is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import io
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
INPUT_NAME = 'Shin_Yamanokami_R1_5_precheck_return.zip'
OUTPUT_NAME = 'Shin_Yamanokami_R1_5_events_return.zip'
DEFAULT_ROOT = Path('data')
STEP_NS = 600_000_000_000
WINDOW_STEPS = 18
STD_MIN = 0.50
DROP_MIN = 0.25
EXPECTED_DEPTHS = {
    'shin': np.arange(25, 346, 20, dtype=float) / 100,
    'yamanokami': np.arange(20, 221, 20, dtype=float) / 100,
}
EVENT_COLUMNS = [
    'event_id', 'catalog', 'site', 'depth_set', 'n_fixed_depths',
    'start_index', 'peak_index', 'min_index', 'end_index',
    'start_index_in_precheck', 'end_index_in_precheck',
    'start_time', 'peak_time', 'min_time', 'end_time', 'window_minutes',
    'STD_start_degC', 'STD_peak_degC', 'STD_min_degC', 'drop_degC',
    'peak_to_min_minutes', 'valid_layers_min', 'valid_layers_max',
    'valid_depth_set_change_count', 'missing_timestamp_rows',
    'nonfinite_temperature_cells', 'QC_unknown_finite_cells',
    'QC_flagged_finite_cells', 'full_3h_window',
    'all_selected_depths_quality_valid', 'strict_window_eligible',
    'not_eligible_reason',
]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def save_csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def row_std(values):
    # Exactly the legacy event diagnostic: population SD, skip missing depths.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanstd(values, axis=1, ddof=0)


def forward_complete(profile_complete):
    profile_complete = np.asarray(profile_complete, dtype=bool)
    n, width = len(profile_complete), WINDOW_STEPS + 1
    out = np.zeros(n, dtype=bool)
    if n >= width:
        c = np.r_[0, np.cumsum(profile_complete.astype(np.int64))]
        out[:n - width + 1] = (c[width:] - c[:-width]) == width
    return out


def candidates(sd, eligible=None):
    # Same thresholds, NaN handling, extremum order and truncated baseline tail
    # as the previously supplied Shiode step1A / original Figure notebook.
    sd = np.asarray(sd, dtype=float)
    out = np.zeros(len(sd), dtype=bool)
    if eligible is not None:
        require(np.shape(eligible) == sd.shape, 'Eligibility length mismatch')
    for i in range(len(sd)):
        if eligible is not None and not eligible[i]:
            continue
        if not np.isfinite(sd[i]) or sd[i] < STD_MIN:
            continue
        w = sd[i:min(i + WINDOW_STEPS + 1, len(sd))]
        if np.isfinite(w).sum() < 2:
            continue
        peak, trough = int(np.nanargmax(w)), int(np.nanargmin(w))
        drop = float(w[peak] - w[trough])
        out[i] = bool(np.isfinite(drop) and drop >= DROP_MIN and trough > peak)
    return out


def merge_starts(candidate):
    starts, last = [], -10**12
    for value in np.flatnonzero(candidate):
        i = int(value)
        if i > last + WINDOW_STEPS:
            starts.append(i)
            last = i
    return starts


def gmean_adjacent(values, depths):
    """No interpolation. Adjacent valid nominal pairs; >=70% valid depths."""
    finite = np.isfinite(values)
    pair = finite[:, :-1] & finite[:, 1:]
    grad = np.abs(np.diff(values, axis=1) / np.diff(depths)[None, :])
    den = pair.sum(axis=1)
    out = np.full(len(values), np.nan)
    ok = (finite.mean(axis=1) >= 0.7) & (den > 0)
    num = np.where(pair, grad, 0).sum(axis=1)
    out[ok] = num[ok] / den[ok]
    return out


def find_input(root, explicit=None):
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        require(p.is_file(), 'Input ZIP does not exist: ' + str(p))
        return p
    here = Path(__file__).resolve().parent
    possible = [here / 'data' / INPUT_NAME, root / INPUT_NAME]
    previous = root / 'revision_R1_5'
    if previous.is_dir():
        possible.extend(previous.glob('shin_yamanokami_precheck_*/' + INPUT_NAME))
    found = sorted(set(p.resolve() for p in possible if p.is_file()))
    require(found, 'Cannot locate ' + INPUT_NAME +
            '. Copy it into this package/data, or pass --input "full_path".')
    digests = {sha256(p) for p in found}
    require(len(digests) == 1,
            'Several DIFFERENT precheck ZIPs found. Specify --input explicitly:\n' +
            '\n'.join(str(p) for p in found))
    return found[0]


def read_site(z, site):
    names = z.namelist()
    for f in (f'{site}/summary.json', f'{site}/observations_for_followup.npz'):
        require(names.count(f) == 1, 'Missing or duplicate archive member: ' + f)
    meta = json.loads(z.read(f'{site}/summary.json').decode('utf-8-sig'))
    with np.load(io.BytesIO(z.read(f'{site}/observations_for_followup.npz')),
                 allow_pickle=False) as a:
        required = ('time_ns', 'depths', 'temperature_raw', 'valid_data_only',
                    'source_row_present', 'required_qc_unknown',
                    'required_qc_bad', 'joint_time_mask')
        require(set(required).issubset(a.files), 'Precheck array schema mismatch')
        arrays = {k: a[k].copy() for k in required}
    t, depth, raw = arrays['time_ns'], arrays['depths'], arrays['temperature_raw']
    require(t.ndim == 1 and t.dtype.kind in 'iu', 'time_ns must be an integer vector')
    require(len(t) >= 19 and np.all(np.diff(t) == STEP_NS),
            'Time axis is not continuous 10-minute data; do not delete missing rows')
    require(raw.shape == (len(t), len(depth)), 'Temperature shape mismatch')
    require(np.shape(depth) == EXPECTED_DEPTHS[site].shape and
            np.allclose(depth, EXPECTED_DEPTHS[site], atol=1e-8, rtol=0),
            'Unexpected nominal depth set: ' + site)
    for k in ('valid_data_only', 'required_qc_unknown', 'required_qc_bad'):
        require(arrays[k].shape == raw.shape and arrays[k].dtype == np.dtype(bool),
                'Bad boolean array: ' + k)
    for k in ('source_row_present', 'joint_time_mask'):
        require(arrays[k].shape == t.shape and arrays[k].dtype == np.dtype(bool),
                'Bad time mask: ' + k)
    finite = np.isfinite(raw)
    expected_valid = (finite & arrays['source_row_present'][:, None] &
                      ~arrays['required_qc_unknown'] & ~arrays['required_qc_bad'])
    require(np.array_equal(expected_valid, arrays['valid_data_only']),
            'Saved QC mask is internally inconsistent; stop rather than alter it')
    idx = np.flatnonzero(arrays['joint_time_mask'])
    require(len(idx) >= 19 and np.all(np.diff(idx) == 1),
            'Joint evaluation span must be one continuous interval')
    scope = meta['scopes']['joint_evaluation_span']
    require(int(t[idx[0]]) == pd.Timestamp(scope['start']).value and
            int(t[idx[-1]]) == pd.Timestamp(scope['end']).value,
            'Time metadata do not match arrays')
    d = {
        'site': site, 'meta': meta, 'time': pd.to_datetime(t[idx]),
        'source_index': idx, 'depths': depth, 'raw': raw[idx],
        'valid': arrays['valid_data_only'][idx],
        'present': arrays['source_row_present'][idx],
        'unknown': arrays['required_qc_unknown'][idx],
        'bad': arrays['required_qc_bad'][idx],
    }
    require(int(d['valid'].all(axis=1).sum()) ==
            int(scope['n_complete_data_quality_profiles']), 'Profile count differs from precheck')
    return d


def take_depths(data, selected):
    out = dict(data)
    for key in ('raw', 'valid', 'unknown', 'bad'):
        out[key] = data[key][:, selected]
    out['depths'] = data['depths'][selected]
    return out


def make_catalog(starts, sd, label, d, depth_set):
    rows = []
    for number, i in enumerate(starts, 1):
        j = min(i + WINDOW_STEPS, len(sd) - 1)
        w = sd[i:j + 1]
        p, q = i + int(np.nanargmax(w)), i + int(np.nanargmin(w))
        finite, valid = np.isfinite(d['raw'][i:j + 1]), d['valid'][i:j + 1]
        full = j - i == WINDOW_STEPS
        missing_rows = int((~d['present'][i:j + 1]).sum())
        missing_cells = int((~finite).sum())
        unknown = int((finite & d['unknown'][i:j + 1]).sum())
        flagged = int((finite & d['bad'][i:j + 1]).sum())
        reasons = []
        for flag, text in [(not full, 'tail_less_than_3h'),
                           (missing_rows, 'missing_timestamp_rows'),
                           (missing_cells, 'nonfinite_temperature'),
                           (unknown, 'unknown_QC'), (flagged, 'data_quality_flagged')]:
            if flag:
                reasons.append(text)
        nvalid = valid.sum(axis=1)
        rows.append(dict(
            event_id=f'{label}_{number:04d}', catalog=label, site=d['site'],
            depth_set=depth_set, n_fixed_depths=len(d['depths']),
            start_index=i, peak_index=p, min_index=q, end_index=j,
            start_index_in_precheck=int(d['source_index'][i]),
            end_index_in_precheck=int(d['source_index'][j]),
            start_time=str(d['time'][i]), peak_time=str(d['time'][p]),
            min_time=str(d['time'][q]), end_time=str(d['time'][j]),
            window_minutes=(j-i)*10, STD_start_degC=float(sd[i]),
            STD_peak_degC=float(sd[p]), STD_min_degC=float(sd[q]),
            drop_degC=float(sd[p]-sd[q]), peak_to_min_minutes=(q-p)*10,
            valid_layers_min=int(nvalid.min()), valid_layers_max=int(nvalid.max()),
            valid_depth_set_change_count=int(np.any(valid[1:] != valid[:-1], axis=1).sum()),
            missing_timestamp_rows=missing_rows, nonfinite_temperature_cells=missing_cells,
            QC_unknown_finite_cells=unknown, QC_flagged_finite_cells=flagged,
            full_3h_window=full, all_selected_depths_quality_valid=bool(valid.all()),
            strict_window_eligible=bool(full and valid.all()),
            not_eligible_reason=';'.join(reasons) if reasons else 'eligible',
        ))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def audit_windows(table, strict_table, strict_candidates):
    out = table.copy()
    exact = {int(r.start_index): r.event_id for r in strict_table.itertuples(index=False)}
    matches, overlaps, still_candidates = [], [], []
    for r in table.itertuples(index=False):
        matches.append(exact.get(int(r.start_index), ''))
        ids = [s.event_id for s in strict_table.itertuples(index=False)
               if s.start_index <= r.end_index and s.end_index >= r.start_index]
        overlaps.append(';'.join(ids))
        still_candidates.append(bool(strict_candidates[int(r.start_index)]))
    out['strict_event_same_start'] = matches
    out['strict_events_overlapping_window'] = overlaps
    out['same_start_is_strict_candidate_before_merge'] = still_candidates
    return out


def month_coverage(time, complete, full3):
    rows = []
    possible = np.arange(len(time)) + WINDOW_STEPS < len(time)
    months = time.strftime('%Y-%m')
    for m in sorted(set(months)):
        use = np.asarray(months == m)
        rows.append(dict(month=m, n_times=int(use.sum()),
                         complete_profiles=int(complete[use].sum()),
                         complete_profile_fraction=float(complete[use].mean()),
                         full_duration_starts=int(possible[use].sum()),
                         complete_3h_starts=int(full3[use].sum())))
    return pd.DataFrame(rows)


def evaluate_depth_set(d, directory, tag):
    directory.mkdir(parents=True, exist_ok=False)
    raw = d['raw']
    masked = np.where(d['valid'], raw, np.nan)
    sr, sa = row_std(raw), row_std(masked)
    complete = d['valid'].all(axis=1)
    full3 = forward_complete(complete)
    sb = np.where(complete, sa, np.nan)
    defs = {'R0': (sr, None), 'A': (sa, None), 'B': (sb, full3)}
    tables, flags, rows = {}, {}, []
    for name, (sd, gate) in defs.items():
        flag = candidates(sd, gate)
        table = make_catalog(merge_starts(flag), sd, name, d, tag)
        save_csv(directory / f'events_{name}.csv', table)
        tables[name], flags[name] = table, flag
        rows.append(dict(site=d['site'], depth_set=tag, catalog=name,
                         n_events=len(table), candidate_starts=int(flag.sum()),
                         reference_event_windows_complete=int(table['strict_window_eligible'].sum())))
    for name in ('R0', 'A'):
        audit = audit_windows(tables[name], tables['B'], flags['B'])
        save_csv(directory / f'audit_{name}_windows_vs_B.csv', audit)
    g = gmean_adjacent(masked, d['depths'])
    series = pd.DataFrame(dict(time=d['time'].astype(str),
        valid_depth_count=d['valid'].sum(axis=1), complete_profile=complete,
        complete_3h_start=full3, STD_R0_raw=sr, STD_A_event_available=sa,
        STD_B_complete=sb, Gmean_A_adjacent_no_interp=g,
        Gmean_B_complete=np.where(complete, g, np.nan)))
    for name, flag in flags.items():
        series[f'candidate_{name}'] = flag
    series.to_csv(directory / 'observation_series.csv.gz', index=False,
                  encoding='utf-8-sig', compression='gzip')
    save_csv(directory / 'coverage_by_month.csv', month_coverage(d['time'], complete, full3))
    np.savez_compressed(directory / 'observations_and_masks.npz',
        time_ns=d['time'].asi8, source_indices=d['source_index'], depths=d['depths'],
        temperature_raw=raw, valid_data_only=d['valid'],
        source_row_present=d['present'], complete_profile=complete,
        complete_3h_start=full3, STD_raw=sr, STD_data_only=sa,
        STD_complete=sb, candidate_R0=flags['R0'], candidate_A=flags['A'],
        candidate_B=flags['B'])
    summary = dict(site=d['site'], depth_set=tag, nominal_depths_m=d['depths'].tolist(),
        start=str(d['time'][0]), end=str(d['time'][-1]), n_times=len(d['time']),
        complete_profiles=int(complete.sum()), complete_3h_starts=int(full3.sum()),
        n_R0_events=len(tables['R0']), n_A_events=len(tables['A']), n_B_events=len(tables['B']),
        R0_original_windows_complete=int(tables['R0']['strict_window_eligible'].sum()),
        A_original_windows_complete=int(tables['A']['strict_window_eligible'].sum()),
        no_event_message=('若B为0，只能表述为给定深度、阈值及完整可评价窗口内未检出；'
                          '不能据此认定缺测期间无事件。'),
        comparison='A vs B changes completeness; R0 vs A also changes QC. '
                   'Gmean uses adjacent valid pairs without interpolation.')
    save_json(directory / 'summary.json', summary)
    return summary, tables, full3, sa, rows


def run(input_zip, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summaries, rows, errors = [], [], []
    settings = dict(version=VERSION, input_zip=str(input_zip), input_sha256=sha256(input_zip),
        STD_start_min_degC=STD_MIN, STD_drop_min_degC=DROP_MIN,
        window_steps=WINDOW_STEPS, step_minutes=10, window_samples=19,
        merging='greedy next_start > previous_start + 18',
        incomplete_tail='R0/A: legacy truncation; B: full 3h required',
        primary_scope='joint_evaluation_span from precheck; no model mask recomputed',
        yamanokami_core_depths_m=EXPECTED_DEPTHS['yamanokami'][1:].tolist(),
        core_rationale='0.20m missing from 2024-04-28 23:50; choose remaining '
                       'fixed 0.40-2.20m BEFORE event analysis, based on observation '
                       'availability only. Supplement, not automatic replacement.',
        no_temperature_interpolation=True, no_model_execution=True,
        no_threshold_sensitivity=True, no_model_metrics=True)
    save_json(out / 'settings.json', settings)
    with zipfile.ZipFile(input_zip) as z:
        batch = json.loads(z.read('batch_summary.json').decode('utf-8-sig'))
        require(batch.get('status') == 'completed' and batch.get('n_errors') == 0,
                'Precheck was not complete; inspect RUN_ERRORS.json first')
        for site in ('shin', 'yamanokami'):
            try:
                print('Checking observations:', site, flush=True)
                d = read_site(z, site)
                base = out / site
                base.mkdir()
                tag = 'original17' if site == 'shin' else 'original11'
                summary, table, full3, sa, records = evaluate_depth_set(d, base / tag, tag)
                summaries.append(summary); rows.extend(records)
                print(site, tag, 'R0/A/B:', summary['n_R0_events'],
                      summary['n_A_events'], summary['n_B_events'], flush=True)
                if site == 'yamanokami':
                    core = take_depths(d, np.arange(1, 11))
                    cs, ct, cf, csa, cr = evaluate_depth_set(core, base / 'core10', 'core10')
                    summaries.append(cs); rows.extend(cr)
                    # Same temporal opportunities as original11 B: isolate the
                    # effect of removing 0.20m from the extra seasonal coverage.
                    gate = full3 & cf
                    flag = candidates(np.where(core['valid'].all(axis=1), csa, np.nan), gate)
                    extra = make_catalog(merge_starts(flag), csa, 'D_same_full11_windows', core, 'core10')
                    save_csv(base / 'core10' / 'events_D_same_full11_windows.csv', extra)
                    save_csv(base / 'core10' / 'audit_original11_B_vs_D.csv',
                             audit_windows(table['B'], extra, flag))
                    rows.append(dict(site=site, depth_set='core10', catalog='D_same_full11_windows',
                                     n_events=len(extra), candidate_starts=int(flag.sum()),
                                     reference_event_windows_complete=len(extra)))
                    print(site, 'core10 R0/A/B:', cs['n_R0_events'],
                          cs['n_A_events'], cs['n_B_events'], flush=True)
            except Exception as exc:
                errors.append(dict(site=site, error=str(exc), traceback=traceback.format_exc()))
                print('FAILED:', site, str(exc), file=sys.stderr, flush=True)
    save_json(out / 'RUN_ERRORS.json', errors)
    save_json(out / 'batch_summary.json', dict(status='completed' if not errors else 'partial_failure',
              n_errors=len(errors), no_model_execution=True, summaries=summaries))
    save_csv(out / 'event_count_summary.csv', pd.DataFrame(rows))
    report = [
        '# R1-5 Shin/Yamanokami：观测事件及固定深度对照', '',
        '由用户本地运行生成；不包含模型重训、推理或四模型精度评价。', '',
        '| 站点 | 深度集合 | R0原数值参考 | A同质控可用层 | B全层完整3小时 |',
        '|---|---|---:|---:|---:|',
    ]
    for s in summaries:
        report.append(f"| {s['site']} | {s['depth_set']} | {s['n_R0_events']} | {s['n_A_events']} | {s['n_B_events']} |")
    report += ['', '## 读表顺序',
        '1. Shin看original17/summary.json和audit_A_windows_vs_B.csv：原来无事件的结论是否仍成立，只以实际结果回答。',
        '2. Yamanokami先看original11；这套严格结果仅对完整11层可评价时段成立。',
        '3. core10是固定0.40–2.20 m的补充覆盖诊断，不代表0.20 m至表面的热结构；不替代original11。',
        '4. core10内A/B比较才是相同深度范围的完整性对照；original11 B与core10 B同时改变深度与可评价时段，不可将差异全归因于缺测。',
        '5. core10/events_D_same_full11_windows.csv在原11层完整候选窗口内计算10层指标，用于与original11 B比较深度范围影响。事件重合表仅按窗口重叠，不是独立物理事件的一一匹配。',
        '', '## 限制',
        '- 阈值固定0.50°C和0.25°C、3小时；未依据结果调整阈值或选择深度。',
        '- 严格完整性在候选合并前检查，旧窗口完整的数量不必等于严格重检事件数。',
        '- 不完整不等于虚假事件；无事件不等于缺测期间无变化。3小时候选起点数不是独立事件数。',
        '- QC沿用输入记录（包括flag_stuck），不证明所有被标记数值都是仪器错误。',
        '- 名义深度不作动态水位修正；不填补观测。',
        '- 这里R0在此前共同评价时段内复算；端点可能与旧图原目标时段不同，不能强行要求事件数与原稿相等。',
        '- 只计算观测的STD与无插值Gmean，尚未验证四模型事件幅度结论或模型覆盖。',
        '- 若某站点RUN_ERRORS非空，不能将该站点没有输出理解为零事件。',
        '', f'运行错误数：{len(errors)}。',
    ]
    (out / 'REPORT_zh.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (out / 'code_used.py').write_bytes(Path(__file__).read_bytes())
    archive = out / OUTPUT_NAME
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p != archive:
                z.write(p, p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'Result ZIP failed CRC check')
    print('UPLOAD THIS FILE:', archive.resolve(), flush=True)
    return len(errors)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--input', type=Path, help='Explicit precheck ZIP, otherwise auto-detect')
    p.add_argument('--output-root', type=Path, help='Default: ROOT/revision_R1_5')
    a = p.parse_args(argv)
    source = find_input(a.root, a.input)
    target_root = a.output_root or a.root / 'revision_R1_5'
    target = target_root / ('shin_yamanokami_events_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print('INPUT:', source)
    print('OUTPUT:', target)
    return 1 if run(source, target) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print('ERROR:', type(e).__name__, str(e), file=sys.stderr)
        sys.exit(1)
