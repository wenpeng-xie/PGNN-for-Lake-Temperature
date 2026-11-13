#!/usr/bin/env python3
"""R1-5: check Shin/Yamanokami observation completeness before event reanalysis.

Read-only input, no training/inference/GLM, no event detection or model scoring.
Python >=3.9; numpy and pandas. Uses the same data-only QC rule as Shiode step1A.
The real observations are to be processed by the user on their own device.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "1.0.0"
DEFAULT_ROOT = 'data'
DATA_FLAGS = ("flag_nan", "flag_range", "flag_spike", "flag_stuck")
IGNORED_FLAGS = ("flag_vert", "flag_unstable", "qc_bad")
STEP_MINUTES = 10
WINDOW_MINUTES = 180
WINDOW_SAMPLES = WINDOW_MINUTES // STEP_MINUTES + 1  # 19, including both ends
TEMP_RE = re.compile(r"^(\d+(?:\.\d*)?|\.\d+)\s*m$", re.I)
FLAG_RE = re.compile(
    r"^(\d+(?:\.\d*)?|\.\d+)\s*m_(flag_nan|flag_range|flag_spike|flag_stuck|flag_vert|flag_unstable|qc_bad)$", re.I
)
SITES = {
    "shin": {
        "name": "Shin", "file": "shin_2025_aligned_qc_wide.csv",
        "depths": [(25 + 20 * i) / 100 for i in range(17)],
        "start": "2025-08-15 13:50:00",
        "observation_end": "2025-10-26 09:30:00",
        "joint_end": "2025-10-26 00:00:00",
        "period_note": "目标测试期起点沿用重训记录；四模型共同评价时段止于GLM输出末端。",
    },
    "yamanokami": {
        "name": "Yamanokami", "file": "yamanokami_2024_aligned_qc_wide.csv",
        "depths": [(20 + 20 * i) / 100 for i in range(11)],
        "start": "2024-02-27 10:00:00",
        "observation_end": "2024-10-14 00:00:00",
        "joint_end": "2024-10-13 21:50:00",
        "period_note": "正式目标期起点位于原序列7天预热之后；共同评价止于此前已核实ML覆盖末端。",
    },
}
SCOPE_LABELS = {
    "joint_evaluation_span": "此前四模型共同评价时段（主检查）",
    "observation_target_span": "原目标观测时段（含末尾，仅作覆盖记录）",
}
DECISIONS = {
    "NO_OBSERVATION_MISSINGNESS_CONTRAST_NEEDED": "本时段观测完整，可省略额外的观测缺测A/B对照",
    "FOLLOWUP_FIXED_DEPTH_ANALYSIS_NEEDED": "本时段存在无效或缺失观测，需要进一步固定深度分析",
}
GAP_COLUMNS = [
    "scope", "channel", "start_time", "end_time", "n_time_points",
    "sample_slots_minutes", "start_to_end_minutes", "absent_timestamp_rows",
    "nonfinite_present_cells", "finite_unknown_qc_cells", "finite_known_bad_qc_cells",
    "min_valid_depths", "max_valid_depths", "valid_depth_set_change_count",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def depth_key(value: float) -> int:
    return int(round(float(value) * 1_000_000))


def resolve_columns(headers: list[str], depths: list[float]) -> tuple[str, list[str], dict, dict]:
    """Match numerical depth exactly (format differences only); never nearest-depth map."""
    require(len(headers) == len(set(headers)), "CSV存在重复原始列名；不自动重命名或选取第一列。")
    time_names = [c for c in headers if c.strip().lower() == "time"]
    require(len(time_names) == 1, "CSV必须有唯一的time列。")
    temps: dict[int, str] = {}
    flags: dict[tuple[int, str], str] = {}
    for c in headers:
        m = TEMP_RE.fullmatch(c.strip())
        if m:
            k = depth_key(float(m.group(1)))
            require(k not in temps, f"同一个名义深度存在多个温度列：{c}")
            temps[k] = c
        m = FLAG_RE.fullmatch(c.strip())
        if m:
            key = (depth_key(float(m.group(1))), m.group(2).lower())
            require(key not in flags, f"同一个深度/质量标记存在多列：{c}")
            flags[key] = c
    missing = []
    selected, required, optional = [], {}, {}
    for d in depths:
        k = depth_key(d)
        if k not in temps:
            missing.append(f"温度列 {d:.2f}m")
            continue
        selected.append(temps[k])
        for flag in DATA_FLAGS:
            if (k, flag) not in flags:
                missing.append(f"{d:.2f}m_{flag}")
            else:
                required[(k, flag)] = flags[(k, flag)]
        for flag in IGNORED_FLAGS:
            if (k, flag) in flags:
                optional[(k, flag)] = flags[(k, flag)]
    require(not missing, "必须使用原始质控宽表qc_wide.csv，缺少：" + ", ".join(missing))
    return time_names[0], selected, required, optional


def parse_flag(series: pd.Series, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return bad and known. Missing flags are unknown, NOT automatically good."""
    x = series.astype("string").str.strip().str.lower().replace("", pd.NA)
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
               "1.0": 1.0, "0.0": 0.0, "t": 1.0, "f": 0.0}
    v = x.map(mapping)
    unexpected = x.notna() & v.isna()
    require(not bool(unexpected.any()), f"无法识别的质量标记 {label}: {x[unexpected].unique().tolist()}")
    known = v.notna().to_numpy(dtype=bool)
    bad = v.fillna(0.0).to_numpy(dtype=float) > 0.5
    return bad, known


def aligned_frame(frame: pd.DataFrame, time_column: str, start: str, end: str):
    """Insert missing timestamps; do not interpolate, average duplicates, round or compact."""
    parsed = pd.to_datetime(frame[time_column], errors="raise")
    require(bool(parsed.notna().all()), "time列包含缺失时间；先核实原文件。")
    source_index = pd.DatetimeIndex(parsed)
    require(source_index.tz is None, "时间包含时区；本脚本不自动转换，请先明确时间约定。")
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    dt = pd.Timedelta(minutes=STEP_MINUTES)
    require(begin.tz is None and finish.tz is None and finish > begin, "检查时段必须为递增、不带时区的时间。")
    require((finish.value - begin.value) % dt.value == 0, "时段两端不在同一10分钟网格上。")
    data = frame.drop(columns=[time_column]).copy()
    data.index = source_index
    data = data.loc[(data.index >= begin) & (data.index <= finish)]
    require(len(data) > 0, "请求时段没有任何数据行。")
    duplicates = data.index[data.index.duplicated(keep=False)]
    require(len(duplicates) == 0, "存在重复时间，不自动平均：" + str(duplicates[:6].tolist()))
    require(bool(np.all((data.index.to_numpy(dtype="datetime64[ns]").astype(np.int64) - begin.value) % dt.value == 0)),
            "存在偏离10分钟网格的时间；不自动四舍五入。")
    was_sorted = bool(data.index.is_monotonic_increasing)
    data = data.sort_index()
    time = pd.date_range(begin, finish, freq=dt)
    require(len(time) >= WINDOW_SAMPLES, "检查时段短于3小时。")
    present = np.asarray(time.isin(data.index), dtype=bool)
    meta = {"source_rows_in_target_period": len(data), "source_time_was_sorted": was_sorted,
            "source_first_time_in_target": str(data.index[0]), "source_last_time_in_target": str(data.index[-1])}
    return time, data.reindex(time), present, meta


def forward_all(complete: np.ndarray, length: int = WINDOW_SAMPLES) -> np.ndarray:
    values = np.asarray(complete, dtype=bool)
    out = np.zeros(len(values), dtype=bool)
    require(length >= 1, "窗口必须至少包含一个时刻。")
    if len(values) >= length:
        sums = np.r_[0, np.cumsum(values.astype(np.int64))]
        out[:len(values)-length+1] = (sums[length:] - sums[:-length]) == length
    return out


def true_intervals(condition: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(condition, dtype=bool)
    delta = np.diff(np.r_[False, x, False].astype(np.int8))
    return list(zip(np.flatnonzero(delta == 1).tolist(), (np.flatnonzero(delta == -1)-1).tolist()))


def completeness_summary(time, raw, valid, present, bad, unknown) -> dict:
    n, k = valid.shape
    finite = np.isfinite(raw)
    full = valid.all(axis=1)
    total_windows = max(0, n-WINDOW_SAMPLES+1)
    eligible = forward_all(full)
    starts_with_room = np.arange(n) < total_windows
    changes = np.r_[False, np.any(valid[1:] != valid[:-1], axis=1)]
    counts = valid.sum(axis=1)
    same_count_changes = changes & np.r_[False, counts[1:] == counts[:-1]]
    gap_runs = true_intervals(~full)
    strict_complete = bool(present.all() and valid.all())
    decision = ("NO_OBSERVATION_MISSINGNESS_CONTRAST_NEEDED" if strict_complete
                else "FOLLOWUP_FIXED_DEPTH_ANALYSIS_NEEDED")
    return dict(
        start=str(time[0]), end=str(time[-1]), n_depths=k, n_expected_times=n,
        n_source_rows_present=int(present.sum()), n_absent_timestamp_rows=int((~present).sum()),
        n_expected_temperature_cells=n*k, n_finite_temperature_cells=int(finite.sum()),
        n_valid_temperature_cells=int(valid.sum()),
        n_nonfinite_cells_on_present_rows=int((present[:, None] & ~finite).sum()),
        n_finite_unknown_qc_cells=int((present[:, None] & finite & unknown).sum()),
        n_finite_known_bad_qc_cells=int((present[:, None] & finite & ~unknown & bad).sum()),
        n_complete_raw_profiles=int(finite.all(axis=1).sum()),
        n_complete_data_quality_profiles=int(full.sum()),
        complete_profile_fraction=float(full.mean()),
        n_incomplete_data_quality_profiles=int((~full).sum()),
        n_full_3h_candidate_starts=total_windows,
        n_complete_3h_candidate_starts=int(eligible.sum()),
        n_incomplete_3h_candidate_starts=int((starts_with_room & ~eligible).sum()),
        n_tail_points_without_full_3h=int((~starts_with_room).sum()),
        n_valid_depth_set_changes=int(changes.sum()),
        n_depth_set_changes_with_same_layer_count=int(same_count_changes.sum()),
        longest_incomplete_profile_run_points=max((j-i+1 for i, j in gap_runs), default=0),
        all_nominal_depths_complete=strict_complete,
        decision_code=decision, decision_zh=DECISIONS[decision],
    )


def interval_rows(scope, time, raw, valid, present, bad, unknown, depths) -> tuple[list, list]:
    finite = np.isfinite(raw)
    counts = valid.sum(axis=1)
    profile, layer = [], []
    def row(i, j, channel, sel):
        f, b, u = finite[i:j+1, sel], bad[i:j+1, sel], unknown[i:j+1, sel]
        pres = present[i:j+1, None]
        vv = valid[i:j+1]
        return dict(scope=scope, channel=channel, start_time=str(time[i]), end_time=str(time[j]),
                    n_time_points=j-i+1, sample_slots_minutes=(j-i+1)*STEP_MINUTES,
                    start_to_end_minutes=(j-i)*STEP_MINUTES,
                    absent_timestamp_rows=int((~present[i:j+1]).sum()),
                    nonfinite_present_cells=int((pres & ~f).sum()),
                    finite_unknown_qc_cells=int((pres & f & u).sum()),
                    finite_known_bad_qc_cells=int((pres & f & ~u & b).sum()),
                    min_valid_depths=int(counts[i:j+1].min()), max_valid_depths=int(counts[i:j+1].max()),
                    valid_depth_set_change_count=int(np.any(vv[1:] != vv[:-1], axis=1).sum()))
    for i, j in true_intervals(~valid.all(axis=1)):
        profile.append(row(i, j, "any_fixed_depth_invalid", slice(None)))
    for c, d in enumerate(depths):
        for i, j in true_intervals(~valid[:, c]):
            layer.append(row(i, j, f"{d:.2f}m", slice(c, c+1)))
    return profile, layer


def monthly_rows(scope, time, raw, valid, present) -> list[dict]:
    months = time.to_period("M").astype(str)
    full = valid.all(axis=1)
    eligible = forward_all(full)
    starts_with_room = np.arange(len(time)) <= len(time)-WINDOW_SAMPLES
    rows = []
    for month in sorted(set(months)):
        ix = np.asarray(months == month)
        n = int(ix.sum())
        rows.append(dict(scope=scope, month=month, expected_time_points=n,
                         source_rows_present=int(present[ix].sum()),
                         finite_temperature_cells=int(np.isfinite(raw[ix]).sum()),
                         valid_temperature_cells=int(valid[ix].sum()),
                         complete_profiles=int(full[ix].sum()),
                         complete_profile_fraction=float(full[ix].mean()),
                         full_3h_candidate_starts=int((ix & starts_with_room).sum()),
                         complete_3h_candidate_starts=int((ix & eligible).sum())))
    return rows


def run_site(root: Path, config: dict, out: Path) -> dict:
    source = root / config["file"]
    require(source.is_file(), "找不到观测文件：" + str(source))
    before = source.stat()
    source_hash = sha256(source)
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        headers = next(csv.reader(f))
    depths = np.asarray(config["depths"], dtype=float)
    time_col, temp_cols, required, optional = resolve_columns(headers, config["depths"])
    usecols = list(dict.fromkeys([time_col] + temp_cols + list(required.values()) + list(optional.values())))
    frame = pd.read_csv(source, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    time, data, present, alignment_meta = aligned_frame(frame, time_col, config["start"], config["observation_end"])
    raw = np.column_stack([pd.to_numeric(data[c], errors="raise").to_numpy(dtype=float) for c in temp_cols])
    bad = np.zeros(raw.shape, dtype=bool)
    unknown = np.zeros(raw.shape, dtype=bool)
    parsed_flags = []
    ignored_parse_warnings = []
    for j, d in enumerate(depths):
        k = depth_key(d)
        for f in DATA_FLAGS:
            col = required[(k, f)]
            b, known = parse_flag(data[col], col)
            bad[:, j] |= b
            unknown[:, j] |= ~known
            parsed_flags.append((j, f, col, True, b, known))
        for f in IGNORED_FLAGS:
            col = optional.get((k, f))
            if col is not None:
                try:
                    b, known = parse_flag(data[col], col)
                    parsed_flags.append((j, f, col, False, b, known))
                except ValueError as exc:
                    ignored_parse_warnings.append(str(exc))
    valid = present[:, None] & np.isfinite(raw) & ~bad & ~unknown
    joint_end = pd.Timestamp(config["joint_end"])
    require(joint_end in time, "共同评价时段终点不在既定观测时间轴内。")
    joint_n = int(time.get_loc(joint_end)) + 1
    scope_slices = {"joint_evaluation_span": slice(0, joint_n), "observation_target_span": slice(None)}
    summary = dict(site=config["name"], version=VERSION, source_path=str(source), source_sha256=source_hash,
                   source_bytes=before.st_size, fixed_nominal_depths_m=depths.tolist(),
                   step_minutes=STEP_MINUTES, window_minutes=WINDOW_MINUTES, window_samples=WINDOW_SAMPLES,
                   data_quality_flags_excluded=list(DATA_FLAGS), ignored_physics_flags=list(IGNORED_FLAGS),
                   unknown_required_qc_is_invalid=True, temperature_interpolation=False,
                   timestamp_rounding=False, water_depth_adjustment=False,
                   prior_joint_period_source="Previously returned model-evaluation records; explicit configured interval, not a fresh model coverage audit.",
                   scopes={}, alignment=alignment_meta, ignored_flag_parse_warnings=ignored_parse_warnings,
                   no_event_detection=True, no_model_prediction_check=True,
                   no_training_or_inference=True,
                   limit="Observations only. Finite values with valid flags do not establish raw instrument validity or absence of upstream filling.")
    depth_rows, month_rows, profile_gaps, depth_gaps, qc_rows = [], [], [], [], []
    for scope, sl in scope_slices.items():
        tt, rr, vv, pp, bb, uu = time[sl], raw[sl], valid[sl], present[sl], bad[sl], unknown[sl]
        summary["scopes"][scope] = completeness_summary(tt, rr, vv, pp, bb, uu)
        month_rows.extend(monthly_rows(scope, tt, rr, vv, pp))
        a, b = interval_rows(scope, tt, rr, vv, pp, bb, uu, depths)
        profile_gaps.extend(a); depth_gaps.extend(b)
        for j, d in enumerate(depths):
            f = np.isfinite(rr[:, j])
            runs = true_intervals(~vv[:, j])
            depth_rows.append(dict(scope=scope, nominal_depth_m=float(d), source_temperature_column=temp_cols[j],
                                   expected_times=len(tt), absent_timestamp_rows=int((~pp).sum()),
                                   finite_temperature_count=int(f.sum()), valid_count=int(vv[:, j].sum()),
                                   valid_fraction=float(vv[:, j].mean()),
                                   nonfinite_present_count=int((pp & ~f).sum()),
                                   finite_unknown_qc_count=int((pp & f & uu[:, j]).sum()),
                                   finite_known_bad_qc_count=int((pp & f & ~uu[:, j] & bb[:, j]).sum()),
                                   longest_invalid_run_points=max((e-s+1 for s, e in runs), default=0)))
        for j, flag, col, used, bflag, known in parsed_flags:
            qc_rows.append(dict(scope=scope, nominal_depth_m=float(depths[j]), flag=flag,
                                source_column=col, used_to_exclude=used,
                                flagged_present_rows=int((bflag[sl] & pp).sum()),
                                unknown_present_rows=int((~known[sl] & pp).sum())))

    complete = valid.all(axis=1)
    complete_full_window = forward_all(complete)
    complete_joint_window = np.zeros(len(time), dtype=bool)
    complete_joint_window[:joint_n] = forward_all(complete[:joint_n])
    enough_full = np.arange(len(time)) <= len(time)-WINDOW_SAMPLES
    enough_joint = np.arange(len(time)) <= joint_n-WINDOW_SAMPLES
    n_valid = valid.sum(axis=1)
    changes = np.r_[False, np.any(valid[1:] != valid[:-1], axis=1)]
    ts = pd.DataFrame(dict(time=time, source_row_present=present,
        n_finite_depths=np.isfinite(raw).sum(axis=1), n_valid_depths=n_valid,
        all_fixed_depths_valid=complete, valid_depth_set_changed=changes,
        changed_with_same_depth_count=changes & np.r_[False, n_valid[1:] == n_valid[:-1]],
        in_joint_evaluation_span=np.arange(len(time)) < joint_n,
        full3h_fits_observation_span=enough_full, complete3h_in_observation_span=complete_full_window,
        full3h_fits_joint_span=enough_joint, complete3h_in_joint_span=complete_joint_window,
        finite_unknown_qc_cells=(np.isfinite(raw) & unknown & present[:, None]).sum(axis=1),
        finite_known_bad_qc_cells=(np.isfinite(raw) & ~unknown & bad & present[:, None]).sum(axis=1)))
    for j, d in enumerate(depths):
        ts[f"valid_{d:.2f}m"] = valid[:, j]
    require((before.st_size, before.st_mtime_ns) == (source.stat().st_size, source.stat().st_mtime_ns),
            "输入文件在读取期间发生变化；停止并用未变化的文件重试。")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "coverage_by_depth.csv", pd.DataFrame(depth_rows))
    write_csv(out / "coverage_by_month.csv", pd.DataFrame(month_rows))
    write_csv(out / "QC_flag_summary.csv", pd.DataFrame(qc_rows))
    write_csv(out / "incomplete_profile_intervals.csv", pd.DataFrame(profile_gaps, columns=GAP_COLUMNS))
    write_csv(out / "invalid_depth_intervals.csv", pd.DataFrame(depth_gaps, columns=GAP_COLUMNS))
    ts.to_csv(out / "completeness_timeseries.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    np.savez_compressed(out / "observations_for_followup.npz", time_ns=time.to_numpy(dtype="datetime64[ns]").astype(np.int64),
                        depths=depths, temperature_raw=raw, valid_data_only=valid,
                        source_row_present=present, required_qc_unknown=unknown, required_qc_bad=bad,
                        joint_time_mask=np.arange(len(time)) < joint_n,
                        complete_profile=complete, complete_3h_start=complete_full_window)
    summary["column_mapping"] = {f"{d:.2f}m": c for d, c in zip(depths, temp_cols)}
    write_json(out / "summary.json", summary)
    return summary


def markdown_report(records: list[dict], errors: list[dict]) -> str:
    lines = ["# R1-5：Shin与Yamanokami观测完整性预检查", "",
             "本文件由用户本地运行生成。只检查观测和既有QC标记；没有运行模型、检测事件或计算模型精度。", "",
             "## 主检查：此前四模型共同评价时段", "",
             "| 站点 | 固定层数 | 应有时刻 | 缺失时间行 | 全层有效时刻 | 完整3小时窗口 / 可容纳完整3小时的起点 | 判定 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in records:
        s = r["scopes"]["joint_evaluation_span"]
        lines.append(f"| {r['site']} | {s['n_depths']} | {s['n_expected_times']} | {s['n_absent_timestamp_rows']} | "
                     f"{s['n_complete_data_quality_profiles']} | {s['n_complete_3h_candidate_starts']} / "
                     f"{s['n_full_3h_candidate_starts']} | {s['decision_zh']} |")
    lines += ["", "## 各站点与两套时间范围", ""]
    for r in records:
        lines += [f"### {r['site']}", f"输入：`{r['source_path']}`。", f"SHA256：`{r['source_sha256']}`。",
                  "名义观测深度（m）：" + ", ".join(f"{d:.2f}" for d in r["fixed_nominal_depths_m"]) + "。", ""]
        for scope, s in r["scopes"].items():
            lines += [f"**{SCOPE_LABELS[scope]}**：{s['start']}—{s['end']}。",
                      f"固定全层完整比例：{s['complete_profile_fraction']:.4%}；深度集合改变次数：{s['n_valid_depth_set_changes']}；"
                      f"其中层数相同但具体层改变：{s['n_depth_set_changes_with_same_layer_count']}。",
                      f"有时间行但温度非有限值：{s['n_nonfinite_cells_on_present_rows']}个位置；"
                      f"有温度但QC不明：{s['n_finite_unknown_qc_cells']}个位置；"
                      f"有温度且QC已知不合格：{s['n_finite_known_bad_qc_cells']}个位置。",
                      f"末尾{s['n_tail_points_without_full_3h']}个时刻之后不足3小时，属于边界，不作为数据缺测。", ""]
    if errors:
        lines += ["## 未完成的站点（不能据此判定没有缺测）", ""]
        for e in errors:
            lines.append(f"- {e['site']}: {e['error']}")
    lines += ["", "## 判读与边界", "",
        "1. 只有主检查时段内固定深度全部通过data-only QC且时间连续，才标为可省略额外观测缺测A/B对照。不是仅检查有没有NaN。",
        "2. 有缺测/QC无效值不等于存在假事件；后续才判断其是否影响事件窗口及模型比较。",
        "3. 未采用模型水深或模型预测掩码来缩小观测检查范围，避免先删掉要检查的缺测。名义深度不做动态水位修正。",
        "4. 未插值、填补、平滑、降采样或删除时间行；0摄氏度不会自动被当作缺测。原文件如已上游填补且无对应标记，仅凭这张表无法识别。",
        "5. 3小时窗口含19个10分钟剖面，必须全部有效；这里是候选起点，不是事件数。未运行任何事件阈值或合并规则。",
        "6. 不因flag_vert、flag_unstable或qc_bad总标记单独剔除。必需QC列缺失或出现无法解释的值会报错，不默认为全部有效。",
        "7. 没有检查模型覆盖，不代表四模型能在全部完整观测窗口评价；更新后的STD/Gmean和事件指标仍是后续任务。",
        "8. 保留固定原深度集合；不根据结果自动删掉浅层或最深层。任何核心深度子集须在后续明确依据后另定。",
        "9. 月度表的窗口归到起始月份；窗口可以跨月，但不能越过评价时段终点。",
        "", "## 输出说明", "",
        "- `overall_summary.csv`：两站点、两时段的完整性与是否需要后续缺测对照。",
        "- 各站点`coverage_by_depth.csv`和`coverage_by_month.csv`：逐层、逐月覆盖。",
        "- `incomplete_profile_intervals.csv`、`invalid_depth_intervals.csv`：连续无效时段。",
        "- `completeness_timeseries.csv.gz`：逐时各深度有效性和3小时完整性；无需手动解压供后续脚本读取。",
        "- `observations_for_followup.npz`：本次范围内原始温度及掩码，供后续事件代码使用；不含模型结果。",
        "- `RUN_ERRORS.json`非空时说明部分检查未完成；其余已成功站点结果仍会保留。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--only", choices=list(SITES), default=None, help="Optional: only recheck one site.")
    a = ap.parse_args(argv)
    root = a.root.expanduser().resolve()
    require(root.is_dir(), "观测根目录不存在：" + str(root))
    outroot = a.out_root.expanduser().resolve() if a.out_root else root / "revision_R1_5"
    out = outroot / ("shin_yamanokami_precheck_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    selected = {a.only: SITES[a.only]} if a.only else SITES
    write_json(out / "settings.json", dict(version=VERSION, root=str(root), sites=selected,
        temporal_scope="two explicit observation-only intervals; no model coverage inferred",
        step_minutes=STEP_MINUTES, window_samples=WINDOW_SAMPLES, data_flags=list(DATA_FLAGS),
        ignore_flags=list(IGNORED_FLAGS), no_interpolation=True))
    records, errors = [], []
    for key, cfg in selected.items():
        print(f"[CHECK] {cfg['name']}: {root / cfg['file']}", flush=True)
        try:
            r = run_site(root, cfg, out / key)
            records.append(r)
            s = r["scopes"]["joint_evaluation_span"]
            print(f"[DONE] {cfg['name']}: complete profiles {s['n_complete_data_quality_profiles']}/{s['n_expected_times']}; "
                  f"{s['decision_code']}", flush=True)
        except Exception as exc:
            errors.append(dict(site=cfg["name"], error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
            print(f"[ERROR] {cfg['name']}: {exc}", file=sys.stderr, flush=True)
    rows = []
    for r in records:
        for scope, s in r["scopes"].items():
            rows.append(dict(site=r["site"], scope=scope, **s))
    write_csv(out / "overall_summary.csv", pd.DataFrame(rows) if rows else pd.DataFrame(columns=["site", "scope", "decision_code"]))
    write_json(out / "RUN_ERRORS.json", errors)
    write_json(out / "batch_summary.json", dict(version=VERSION, requested_sites=list(selected),
        completed_sites=[r["site"] for r in records], status="completed" if not errors else "incomplete_check_errors",
        n_errors=len(errors), no_actual_model_execution=True, no_event_detection=True))
    (out / "REPORT_zh.md").write_text(markdown_report(records, errors), encoding="utf-8")
    shutil.copyfile(Path(__file__).resolve(), out / "code_used.py")
    files_to_pack = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = [{"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files_to_pack]
    write_csv(out / "manifest.csv", pd.DataFrame(manifest))
    archive = out / "Shin_Yamanokami_R1_5_precheck_return.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in files_to_pack + [out / "manifest.csv"]:
            mode = zipfile.ZIP_STORED if p.suffix in {".gz", ".npz"} else zipfile.ZIP_DEFLATED
            z.write(p, arcname=p.relative_to(out).as_posix(), compress_type=mode)
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "返回ZIP的CRC检查失败。")
    print("\nREPORT: " + str(out / "REPORT_zh.md"), flush=True)
    print("UPLOAD THIS FILE: " + str(archive), flush=True)
    if errors:
        print("部分站点检查未完成；请上传返回包排查，不要据报错推断没有缺测。", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        print("原始文件未修改。", file=sys.stderr)
        raise SystemExit(1)
