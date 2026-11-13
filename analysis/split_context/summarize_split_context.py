#!/usr/bin/env python3
"""
JoHRS Reviewer 2-5: summarize split, seasonal coverage, and final test-event frequency.

This v3 intentionally avoids forcing historical monitoring periods onto the final
R1-5 fixed-depth grids. The reviewer asked whether different training/testing
strategies, seasonal conditions, data coverage, and event prevalence may confound
cross-site interpretation.

Therefore:
  * Yamanokami/Shiode use the original monitoring-period *_aligned.csv files
    already used for the observation-overview figure.
  * Each period is summarized on its own native temperature sensor grid within
    the manuscript-relevant depth range.
  * Shin uses the corrected *_aligned_qc_wide.csv so the 56/14/30 split indices
    match the retraining audit.
  * Final TEST event counts/frequencies are read from the already verified R1-5
    metadata (Shiode 91, Yamanokami 13, Shin 0). Events are NOT re-detected here.

No model is trained, evaluated, or rerun.

Dependencies: numpy, pandas
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_ZIP = "JoHRS_R2_5_context_metadata.zip"

PERIODS = [
    {
        "site": "Yamanokami",
        "role": "Development/training",
        "year": 2025,
        "filename": "yamanokami_2025_aligned.csv",
        "start": "2025-03-01 00:00:00",
        "end": "2025-11-23 00:00:00",
        "depth_min": 0.20,
        "depth_max": 2.25,
    },
    {
        "site": "Yamanokami",
        "role": "External test",
        "year": 2024,
        "filename": "yamanokami_2024_aligned.csv",
        "start": "2024-02-20 10:00:00",
        "end": "2024-10-14 00:00:00",
        "depth_min": 0.20,
        "depth_max": 2.25,
    },
    {
        "site": "Shiode",
        "role": "Development/training",
        "year": 2024,
        "filename": "shiode_2024_aligned.csv",
        "start": "2024-02-21 00:00:00",
        "end": "2024-11-14 00:00:00",
        "depth_min": 0.20,
        "depth_max": 2.25,
    },
    {
        "site": "Shiode",
        "role": "External test",
        "year": 2025,
        "filename": "shiode_2025_aligned.csv",
        "start": "2025-05-01 00:00:00",
        "end": "2025-11-23 00:00:00",
        "depth_min": 0.20,
        "depth_max": 2.25,
    },
]

MET_CANDIDATES = {
    "solar": ["solar", "shortwave", "swdown", "swrad"],
    "precip": ["precip", "rain", "rainfall"],
    "wind": ["wind", "windspeed", "wind_speed", "ws"],
    "air_temp": ["air_temp", "airtemperature", "tair", "airtemp"],
}


def require(cond, msg):
    if not bool(cond):
        raise ValueError(msg)


def norm(s):
    return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")


def find_column(df, candidates, required=True):
    cn = [norm(x) for x in candidates]
    for col in df.columns:
        c = norm(col)
        if any(c == x or x in c for x in cn):
            return col
    if required:
        raise ValueError(
            f"Cannot find required column among {candidates}; "
            f"available columns={list(df.columns)}"
        )
    return None


def parse_depth_from_col(col):
    """
    Parse temperature sensor depth from columns such as:
      0.2m
      0.2 m
      temp_0.2m
    Excludes QC/flag columns.
    """
    s = str(col).strip().lower()
    if "flag" in s or "qc" in s:
        return None

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*_?\s*m\s*$", s)
    if m:
        return float(m.group(1))

    # Plain numeric label, occasionally produced after earlier preprocessing.
    try:
        v = float(s)
        if 0 < v <= 10:
            return v
    except Exception:
        pass

    return None


def parse_depth_from_flag_col(col):
    s = str(col).strip().lower()
    if "flag" not in s and "qc" not in s:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*_?\s*m", s)
    if m:
        return float(m.group(1))
    return None


def find_file(root, filename):
    hits = sorted(set(p.resolve() for p in root.rglob(filename)))
    require(hits, f"Cannot find {filename} below {root}")

    if len(hits) > 1:
        print(f"[FILE CANDIDATES] {filename}")
        for h in hits:
            print(" ", h)
        # Prefer the shallower path; this tends to select the canonical project-root copy.
        hits = sorted(hits, key=lambda p: (len(p.parts), str(p)))
        print(" -> using", hits[0])

    return hits[0]


def load_period(path, start=None, end=None):
    df = pd.read_csv(path)
    tcol = find_column(df, ["time", "datetime", "date_time"], required=True)
    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = (
        df.dropna(subset=[tcol])
        .sort_values(tcol)
        .drop_duplicates(subset=[tcol], keep="last")
        .reset_index(drop=True)
    )

    if start is not None:
        df = df[df[tcol] >= pd.Timestamp(start)].copy()
    if end is not None:
        df = df[df[tcol] <= pd.Timestamp(end)].copy()

    df = df.reset_index(drop=True)
    require(len(df) > 0, f"No data after date filtering: {path}")
    return df, tcol


def native_temperature_columns(df, depth_min, depth_max):
    parsed = []
    for c in df.columns:
        d = parse_depth_from_col(c)
        if d is None:
            continue
        if float(depth_min) - 1e-9 <= d <= float(depth_max) + 1e-9:
            parsed.append((d, c))

    parsed = sorted(parsed)
    require(
        len(parsed) >= 3,
        f"Fewer than 3 temperature sensor columns found in "
        f"{depth_min}-{depth_max} m. Parsed={parsed}"
    )
    return parsed


def qc_flags_for_sensor(df, sensor_depth):
    """Use only data-quality flags if present; ignore vert/unstable flags."""
    out = []
    for kind in ["nan", "range", "spike", "stuck"]:
        hits = []
        for c in df.columns:
            nc = norm(c)
            if kind not in nc:
                continue
            d = parse_depth_from_flag_col(c)
            if d is not None and abs(d - sensor_depth) <= 1e-6:
                hits.append(c)
        if len(hits) == 1:
            out.append(hits[0])
        elif len(hits) > 1:
            raise ValueError(
                f"Ambiguous QC flags for depth={sensor_depth}, kind={kind}: {hits}"
            )
    return out


def bool_flag(series):
    if series.dtype == bool:
        return series.to_numpy(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).to_numpy(bool)


def extract_native_temp_and_valid(df, parsed_depth_cols):
    depths = np.array([d for d, _ in parsed_depth_cols], dtype=float)
    cols = [c for _, c in parsed_depth_cols]

    temp = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(temp)

    qc_used = False
    n_qc_cols = 0
    for j, d in enumerate(depths):
        flags = qc_flags_for_sensor(df, d)
        if flags:
            qc_used = True
            n_qc_cols += len(flags)
            for fc in flags:
                valid[:, j] &= ~bool_flag(df[fc])

    complete = valid.all(axis=1)
    return depths, cols, temp, valid, complete, qc_used, n_qc_cols


def qstats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.quantile(x, 0.25)),
        float(np.median(x)),
        float(np.quantile(x, 0.75)),
    )


def summarize_period(site, role, path, df, tcol, depth_min, depth_max):
    parsed = native_temperature_columns(df, depth_min, depth_max)
    depths, cols, temp, valid, complete, qc_used, n_qc_cols = (
        extract_native_temp_and_valid(df, parsed)
    )

    print(f"[NATIVE DEPTHS] {site} {role}")
    print(" ", ", ".join(f"{d:.2f}m->{c}" for d, c in parsed))

    time = pd.DatetimeIndex(df[tcol])
    dt_min = (
        float(np.median(np.diff(time.asi8)) / 60e9)
        if len(time) > 1 else np.nan
    )

    profile_mean = np.full(len(df), np.nan)
    profile_std = np.full(len(df), np.nan)
    if complete.any():
        profile_mean[complete] = np.mean(temp[complete], axis=1)
        profile_std[complete] = np.std(temp[complete], axis=1, ddof=0)

    met = {}
    for key, candidates in MET_CANDIDATES.items():
        col = find_column(df, candidates, required=False)
        met[key] = (
            pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            if col is not None
            else np.full(len(df), np.nan)
        )

    air_q25, air_med, air_q75 = qstats(met["air_temp"])
    sol_q25, sol_med, sol_q75 = qstats(met["solar"])
    wind_q25, wind_med, wind_q75 = qstats(met["wind"])
    wt_q25, wt_med, wt_q75 = qstats(profile_mean)
    std_q25, std_med, std_q75 = qstats(profile_std)

    precip = met["precip"]
    finite_precip = np.isfinite(precip)
    precip_occ = (
        float(np.mean(precip[finite_precip] > 0) * 100.0)
        if finite_precip.any() else np.nan
    )

    months = sorted(set(time.month.tolist()))
    span_days = (
        (time[-1] - time[0]).total_seconds() / 86400.0
        if len(time) > 1 else 0.0
    )
    complete_days = (
        float(complete.sum() * dt_min / (60 * 24))
        if np.isfinite(dt_min) else np.nan
    )

    return {
        "site": site,
        "role": role,
        "source_file": str(path),
        "start": str(time[0]),
        "end": str(time[-1]),
        "months_present": ",".join(str(m) for m in months),
        "n_rows": int(len(df)),
        "median_timestep_min": dt_min,
        "calendar_span_days": span_days,
        "selected_native_depths_m": ";".join(f"{d:.2f}" for d in depths),
        "n_selected_native_depths": int(len(depths)),
        "complete_profile_count": int(complete.sum()),
        "complete_profile_fraction": float(complete.mean()),
        "complete_profile_equiv_days": complete_days,
        "quality_basis": (
            "data-only QC flags + finite values"
            if qc_used
            else "finite values only (no matching data-only QC flags in source)"
        ),
        "n_qc_flag_columns_used": int(n_qc_cols),
        "air_temp_q25_C": air_q25,
        "air_temp_median_C": air_med,
        "air_temp_q75_C": air_q75,
        "solar_q25": sol_q25,
        "solar_median": sol_med,
        "solar_q75": sol_q75,
        "wind_q25": wind_q25,
        "wind_median": wind_med,
        "wind_q75": wind_q75,
        "precip_positive_fraction_percent": precip_occ,
        "profile_mean_temp_q25_C": wt_q25,
        "profile_mean_temp_median_C": wt_med,
        "profile_mean_temp_q75_C": wt_q75,
        "native_grid_vertical_STD_q25_C": std_q25,
        "native_grid_vertical_STD_median_C": std_med,
        "native_grid_vertical_STD_q75_C": std_q75,
    }


def load_metadata(meta_zip):
    with zipfile.ZipFile(meta_zip) as z:
        split = json.loads(z.read("shin_data_split_report.json"))
        events = json.loads(z.read("final_test_event_context.json"))
    return split, events


def event_frequency_table(event_context):
    rows = []
    for site in ["Yamanokami", "Shiode", "Shin"]:
        x = event_context[site]
        complete_days = x["complete_profiles"] * 10.0 / (60 * 24)
        rows.append({
            "site": site,
            "final_test_event_count": int(x["final_events"]),
            "final_test_fixed_depths_m": ";".join(
                f"{d:.2f}" for d in x["fixed_depths_m"]
            ),
            "event_analysis_start": x["time_start"],
            "event_analysis_end": x["time_end"],
            "complete_profiles": int(x["complete_profiles"]),
            "eligible_3h_starts": int(x["eligible_3h_starts"]),
            "events_per_1000_eligible_3h_starts": (
                1000.0 * x["final_events"] / x["eligible_3h_starts"]
                if x["eligible_3h_starts"] else np.nan
            ),
            "complete_profile_equiv_days": complete_days,
            "events_per_30_complete_profile_days": (
                30.0 * x["final_events"] / complete_days
                if complete_days > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    require(root.is_dir(), f"Root does not exist: {root}")

    if args.metadata is not None:
        meta = args.metadata.expanduser().resolve()
    else:
        hits = sorted(set(p.resolve() for p in root.rglob(METADATA_ZIP)))
        require(hits, f"Cannot find {METADATA_ZIP} below {root}")
        meta = hits[0]

    split, event_context = load_metadata(meta)

    out = (
        args.out.expanduser().resolve()
        if args.out
        else root / "revision_R2_5" / "split_context_v3"
    )
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    # Cross-year periods from original observation-overview files.
    for cfg in PERIODS:
        path = find_file(root, cfg["filename"])
        df, tcol = load_period(path, cfg["start"], cfg["end"])
        rows.append(
            summarize_period(
                cfg["site"],
                cfg["role"],
                path,
                df,
                tcol,
                cfg["depth_min"],
                cfg["depth_max"],
            )
        )

    # Shin exact corrected 56/14/30 split.
    shin_path = find_file(root, "shin_2025_aligned_qc_wide.csv")
    full, tcol = load_period(shin_path)

    require(
        len(full) == int(split["n_rows"]),
        f"Shin source rows {len(full)} do not match corrected split report "
        f"n_rows={split['n_rows']}"
    )

    train_end = int(split["train_end_exclusive"])
    dev_end = int(split["development_end_exclusive"])
    require(0 < train_end < dev_end < len(full), "Invalid Shin split indices.")

    pieces = [
        ("Train (56%)", full.iloc[:train_end].copy()),
        ("Validation (14%)", full.iloc[train_end:dev_end].copy()),
        ("Development (70%)", full.iloc[:dev_end].copy()),
        ("Test (30%)", full.iloc[dev_end:].copy()),
    ]

    for role, df in pieces:
        df = df.reset_index(drop=True)
        rows.append(
            summarize_period(
                "Shin",
                role,
                shin_path,
                df,
                tcol,
                0.25,
                3.45,
            )
        )

    period_df = pd.DataFrame(rows)
    period_df.to_csv(
        out / "Table_R2_5_period_context.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_df = pd.DataFrame([
        {
            "site": "Yamanokami",
            "development_or_train": "2025 monitoring period",
            "validation": "internal validation within development year",
            "external_test": "2024 monitoring period",
            "strategy": "cross-year",
            "interpretation": (
                "interannual generalization; training and test seasonal/thermal "
                "distributions are not identical"
            ),
        },
        {
            "site": "Shiode",
            "development_or_train": "2024 monitoring period",
            "validation": "internal validation within development year",
            "external_test": "2025 monitoring period",
            "strategy": "cross-year reversed",
            "interpretation": (
                "interannual generalization; final 2025 test period is strongly "
                "event-rich under the corrected diagnostic"
            ),
        },
        {
            "site": "Shin",
            "development_or_train": (
                f"rows 0:{dev_end}; train rows 0:{train_end}, "
                f"validation rows {train_end}:{dev_end}"
            ),
            "validation": "14% internal chronological validation",
            "external_test": f"rows {dev_end}:{len(full)}",
            "strategy": "within-year chronological 56/14/30",
            "interpretation": (
                "later-season within-year generalization rather than interannual transfer"
            ),
        },
    ])
    split_df.to_csv(
        out / "Table_R2_5_split_design.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_df = event_frequency_table(event_context)
    event_df.to_csv(
        out / "Table_R2_5_final_test_event_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# Reviewer 2-5 split / season / coverage context",
        "",
        "No model was retrained or re-evaluated.",
        "",
        "## Why native depth grids are used here",
        "",
        "This analysis is contextual, not a new event detector. Historical monitoring periods",
        "used different sensor grids, so v3 reports each period on its own native sensor grid",
        "and records those depths explicitly. The final TEST event-frequency comparison remains",
        "based on the already verified fixed-depth R1-5 catalogs.",
        "",
        "Do not interpret native-grid vertical-STD differences as an exact apples-to-apples",
        "structural comparison across years; they are descriptive context only.",
        "",
        "## Final corrected TEST-event frequency",
        "",
    ]

    for r in event_df.itertuples(index=False):
        lines += [
            f"### {r.site}",
            f"- final events: {r.final_test_event_count}",
            f"- eligible complete 3-h starts: {r.eligible_3h_starts}",
            f"- events per 1000 eligible starts: {r.events_per_1000_eligible_3h_starts:.3f}",
            f"- events per 30 complete-profile days: {r.events_per_30_complete_profile_days:.3f}",
            "",
        ]

    lines += [
        "## Manuscript interpretation",
        "",
        "- Cross-site performance differences should not be attributed to reservoir physics alone.",
        "- Yamanokami and Shiode evaluate cross-year generalization under different seasonal/interannual distributions.",
        "- Shin evaluates later-season within-year generalization using the corrected 56/14/30 split.",
        "- Training-data availability and final test-event prevalence differ substantially among sites and should be treated as confounders.",
        "- The final corrected test event counts are 13 (Yamanokami), 91 (Shiode), and 0 (Shin).",
        "",
        "A compact table plus Discussion is sufficient for Reviewer 2-5; no new model training is implied.",
    ]

    (out / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")

    ret = out / "JoHRS_R2_5_split_context_return.zip"
    with zipfile.ZipFile(ret, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(out.iterdir()):
            if fp.is_file() and fp != ret:
                z.write(fp, fp.name)

    print("\nCOMPLETE")
    print(period_df[[
        "site", "role", "start", "end",
        "n_rows", "n_selected_native_depths",
        "complete_profile_fraction",
        "air_temp_median_C",
        "profile_mean_temp_median_C",
    ]].to_string(index=False))
    print("\nUPLOAD:")
    print(ret)


if __name__ == "__main__":
    main()
