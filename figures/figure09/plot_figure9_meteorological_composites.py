#!/usr/bin/env python3
"""Rebuild revised JoHRS Figure 9 from final fixed-depth event catalogs.

Default analysis:
- final observation-defined event catalogs: Yamanokami 13, Shiode 91
- align at observed STD(t) peak (t=0)
- window: -6 to +6 h
- five matched non-event controls per retained event
- controls: same month and same local time when possible; fallback to same local time
- controls excluded within +/-12 h of any final event peak
- continuous variables: median and interquartile range
- precipitation: mean amount (not occurrence percentage)
- outputs PNG, PDF, editable-text SVG, CSV audits, and provenance JSON
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_ORDER = ["shiode", "yamanokami"]
SITE_LABELS = {"shiode": "Shiode", "yamanokami": "Yamanokami"}
EVENT_COLOR = "#1f77b4"
CONTROL_COLOR = "#e49c3f"

PLOT_VARIABLES = [
    ("stdT", r"Observed STD(t) ($^\circ$C)", "median_iqr"),
    ("solar", r"Solar radiation (W m$^{-2}$)", "median_iqr"),
    ("wind", r"Wind speed (m s$^{-1}$)", "median_iqr"),
    ("air_temp", r"Air temperature ($^\circ$C)", "median_iqr"),
    ("precip", "Mean precipitation (mm per 10 min)", "mean"),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def find_one(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


def robust_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def read_csv_direct_or_tar(root: Path, csv_name: str, tar_name: str) -> Tuple[pd.DataFrame, str]:
    direct_matches = sorted(root.rglob(csv_name))
    if direct_matches:
        path = direct_matches[0]
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig"), str(path)

    tar_path = find_one(root, tar_name)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [n for n in tf.getnames() if n.endswith(csv_name)]
        require(len(members) == 1, f"Expected exactly one {csv_name} in {tar_path}; found {members}")
        fileobj = tf.extractfile(members[0])
        require(fileobj is not None, f"Could not extract {members[0]}")
        df = pd.read_csv(fileobj, sep=None, engine="python", encoding="utf-8-sig")
        return df, f"{tar_path}::{members[0]}"


def clean_meteorology(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip().lower() for c in out.columns]
    required = ["time", "solar", "precip", "wind", "air_temp"]
    missing = [c for c in required if c not in out.columns]
    require(not missing, f"Meteorology source {source} is missing columns {missing}; columns={list(out.columns)}")
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    for c in ["solar", "precip", "wind", "air_temp", "rh", "pressure"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_yamanokami_catalog(event_zip: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(event_zip) as zf:
        events = pd.read_csv(io.BytesIO(zf.read("yamanokami/core10/events_B.csv")))
        obs = pd.read_csv(io.BytesIO(zf.read("yamanokami/core10/observation_series.csv.gz")), compression="gzip")
    events = events.copy()
    for c in ["start_time", "peak_time", "min_time", "end_time"]:
        events[c] = pd.to_datetime(events[c])
    obs["time"] = pd.to_datetime(obs["time"])
    obs = obs[["time", "STD_B_complete"]].rename(columns={"STD_B_complete": "stdT"})
    return events, obs


def load_shiode_catalog(step1b_zip: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(step1b_zip) as zf:
        events = pd.read_csv(io.BytesIO(zf.read("input_events_B_fixed11_complete3h.csv")))
        npz = np.load(io.BytesIO(zf.read("ensemble_structure_series.npz")))
        obs = pd.DataFrame({
            "time": pd.to_datetime(npz["time"].astype("datetime64[ns]")),
            "stdT": np.asarray(npz["obs_STD_fixed11"], dtype=float),
        })
    for c in ["start_time", "peak_time", "min_time", "end_time"]:
        events[c] = pd.to_datetime(events[c])
    return events, obs


def infer_dt_minutes(time: pd.DatetimeIndex) -> int:
    diffs = pd.Series(time).diff().dropna().dt.total_seconds().to_numpy() / 60.0
    require(len(diffs) > 0, "Need at least two timestamps")
    dt = int(round(float(np.nanmedian(diffs))))
    require(dt > 0, f"Invalid time step {dt}")
    return dt


def align_observation_and_meteorology(obs: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    obs = obs.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    met = met.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    dt = infer_dt_minutes(pd.DatetimeIndex(obs["time"]))
    tolerance = pd.Timedelta(minutes=max(1.0, 0.51 * dt))
    aligned = pd.merge_asof(
        obs,
        met[["time", "solar", "precip", "wind", "air_temp"]],
        on="time",
        direction="nearest",
        tolerance=tolerance,
    )
    return aligned


def map_event_peaks_to_indices(events: pd.DataFrame, aligned: pd.DataFrame, tolerance_minutes: float = 5.1) -> pd.DataFrame:
    time = pd.DatetimeIndex(aligned["time"])
    out = events.copy().sort_values("peak_time").reset_index(drop=True)
    mapped = []
    differences = []
    for peak in out["peak_time"]:
        pos = time.get_indexer([pd.Timestamp(peak)], method="nearest")[0]
        require(pos >= 0, f"Could not map event peak {peak}")
        diff = abs((time[pos] - pd.Timestamp(peak)).total_seconds()) / 60.0
        require(diff <= tolerance_minutes, f"Event peak {peak} differs from nearest reference time by {diff:.2f} min")
        mapped.append(int(pos))
        differences.append(float(diff))
    out["peak_idx_composite"] = mapped
    out["peak_time_mapping_difference_min"] = differences
    return out


def extract_window(values: np.ndarray, center_idx: int, before_steps: int, after_steps: int) -> np.ndarray | None:
    i0 = center_idx - before_steps
    i1 = center_idx + after_steps
    if i0 < 0 or i1 >= len(values):
        return None
    return np.asarray(values[i0:i1 + 1], dtype=float)


def valid_window(aligned: pd.DataFrame, center_idx: int, before_steps: int, after_steps: int, min_valid_fraction: float) -> bool:
    for variable in ["stdT", "solar", "wind", "air_temp", "precip"]:
        window = extract_window(aligned[variable].to_numpy(dtype=float), center_idx, before_steps, after_steps)
        if window is None or np.isfinite(window).mean() < min_valid_fraction:
            return False
    return True


def choose_controls(
    aligned: pd.DataFrame,
    events_all: pd.DataFrame,
    events_used: pd.DataFrame,
    before_steps: int,
    after_steps: int,
    exclusion_steps: int,
    n_controls: int,
    min_valid_fraction: float,
) -> pd.DataFrame:
    time = pd.DatetimeIndex(aligned["time"])
    n = len(aligned)
    blocked = np.zeros(n, dtype=bool)
    for peak_idx in events_all["peak_idx_composite"].astype(int):
        i0 = max(0, peak_idx - exclusion_steps)
        i1 = min(n, peak_idx + exclusion_steps + 1)
        blocked[i0:i1] = True

    valid_center = np.zeros(n, dtype=bool)
    for idx in range(before_steps, n - after_steps):
        if not blocked[idx] and valid_window(aligned, idx, before_steps, after_steps, min_valid_fraction):
            valid_center[idx] = True

    minute_of_day = time.hour * 60 + time.minute
    rows: List[dict] = []
    for _, event in events_used.iterrows():
        peak_idx = int(event["peak_idx_composite"])
        peak_time = time[peak_idx]
        minute = peak_time.hour * 60 + peak_time.minute

        primary = valid_center & (time.month == peak_time.month) & (minute_of_day == minute)
        candidates = np.flatnonzero(primary)
        match_level = {int(i): "same_month_same_local_time" for i in candidates}

        if len(candidates) < n_controls:
            fallback = valid_center & (minute_of_day == minute)
            fallback_candidates = np.flatnonzero(fallback)
            for i in fallback_candidates:
                match_level.setdefault(int(i), "same_local_time_any_month")
            candidates = np.unique(np.concatenate([candidates, fallback_candidates]))

        require(len(candidates) > 0, f"No matched controls for event {event['event_id']} at {peak_time}")
        distance = np.abs((time[candidates] - peak_time).total_seconds())
        selected = candidates[np.argsort(distance)[:n_controls]]
        for rank, idx in enumerate(selected, 1):
            rows.append({
                "site": str(event["site"]),
                "event_id": str(event["event_id"]),
                "event_peak_time": peak_time,
                "control_rank": rank,
                "control_idx": int(idx),
                "control_time": time[idx],
                "match_level": match_level[int(idx)],
                "absolute_time_distance_days": abs((time[idx] - peak_time).total_seconds()) / 86400.0,
            })
    return pd.DataFrame(rows)


def stack_windows(aligned: pd.DataFrame, centers: Iterable[int], column: str, before_steps: int, after_steps: int) -> np.ndarray:
    windows = []
    values = aligned[column].to_numpy(dtype=float)
    for idx in centers:
        window = extract_window(values, int(idx), before_steps, after_steps)
        if window is not None:
            windows.append(window)
    require(len(windows) > 0, f"No windows for {column}")
    return np.stack(windows, axis=0)


def median_iqr(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmedian(matrix, axis=0),
        np.nanpercentile(matrix, 25, axis=0),
        np.nanpercentile(matrix, 75, axis=0),
    )


def prepare_site(
    site: str,
    events: pd.DataFrame,
    obs: pd.DataFrame,
    met: pd.DataFrame,
    before_hours: float,
    after_hours: float,
    n_controls: int,
    exclusion_hours: float,
    min_valid_fraction: float,
) -> dict:
    aligned = align_observation_and_meteorology(obs, met)
    dt_minutes = infer_dt_minutes(pd.DatetimeIndex(aligned["time"]))
    steps_per_hour = 60.0 / dt_minutes
    before_steps = int(round(before_hours * steps_per_hour))
    after_steps = int(round(after_hours * steps_per_hour))
    exclusion_steps = int(round(exclusion_hours * steps_per_hour))

    events = events.copy()
    events["site"] = site
    events = map_event_peaks_to_indices(events, aligned)
    valid_mask = events["peak_idx_composite"].apply(
        lambda idx: valid_window(aligned, int(idx), before_steps, after_steps, min_valid_fraction)
    )
    used = events.loc[valid_mask].copy().reset_index(drop=True)
    require(len(used) > 0, f"{site}: no events remain after +/-{before_hours} h composite-window screening")
    controls = choose_controls(
        aligned=aligned,
        events_all=events,
        events_used=used,
        before_steps=before_steps,
        after_steps=after_steps,
        exclusion_steps=exclusion_steps,
        n_controls=n_controls,
        min_valid_fraction=min_valid_fraction,
    )
    return {
        "site": site,
        "aligned": aligned,
        "events_all": events,
        "events_used": used,
        "controls": controls,
        "dt_minutes": dt_minutes,
        "steps_per_hour": steps_per_hour,
        "before_steps": before_steps,
        "after_steps": after_steps,
    }


def plot_and_export(site_results: Dict[str, dict], outdir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(11.6, 13.4), sharex=True, squeeze=False)
    output_rows: List[dict] = []

    for col, site in enumerate(SITE_ORDER):
        result = site_results[site]
        aligned = result["aligned"]
        events = result["events_used"]
        controls = result["controls"]
        before_steps = result["before_steps"]
        after_steps = result["after_steps"]
        steps_per_hour = result["steps_per_hour"]
        event_centers = events["peak_idx_composite"].astype(int).to_numpy()
        control_centers = controls["control_idx"].astype(int).to_numpy()
        relative_hours = np.arange(-before_steps, after_steps + 1) / steps_per_hour

        for row, (variable, ylabel, mode) in enumerate(PLOT_VARIABLES):
            ax = axes[row, col]
            event_matrix = stack_windows(aligned, event_centers, variable, before_steps, after_steps)
            control_matrix = stack_windows(aligned, control_centers, variable, before_steps, after_steps)
            n_event_valid = np.sum(np.isfinite(event_matrix), axis=0)
            n_control_valid = np.sum(np.isfinite(control_matrix), axis=0)

            if mode == "median_iqr":
                e_mid, e_q25, e_q75 = median_iqr(event_matrix)
                c_mid, c_q25, c_q75 = median_iqr(control_matrix)
                ax.plot(relative_hours, e_mid, color=EVENT_COLOR, linewidth=2.0, label="Events")
                ax.fill_between(relative_hours, e_q25, e_q75, color=EVENT_COLOR, alpha=0.18, linewidth=0)
                ax.plot(relative_hours, c_mid, color=CONTROL_COLOR, linestyle="--", linewidth=1.8, label="Matched non-event controls")
                ax.fill_between(relative_hours, c_q25, c_q75, color=CONTROL_COLOR, alpha=0.12, linewidth=0)
                for i, hr in enumerate(relative_hours):
                    output_rows.append({
                        "site": site, "variable": variable, "relative_hour": hr, "summary": "median_iqr",
                        "event_center": e_mid[i], "event_q25": e_q25[i], "event_q75": e_q75[i],
                        "control_center": c_mid[i], "control_q25": c_q25[i], "control_q75": c_q75[i],
                        "n_event_valid": int(n_event_valid[i]), "n_control_valid": int(n_control_valid[i]),
                    })
            else:
                e_mean = np.nanmean(event_matrix, axis=0)
                c_mean = np.nanmean(control_matrix, axis=0)
                ax.plot(relative_hours, e_mean, color=EVENT_COLOR, linewidth=2.0, label="Events")
                ax.plot(relative_hours, c_mean, color=CONTROL_COLOR, linestyle="--", linewidth=1.8, label="Matched non-event controls")
                for i, hr in enumerate(relative_hours):
                    output_rows.append({
                        "site": site, "variable": variable, "relative_hour": hr, "summary": "mean",
                        "event_center": e_mean[i], "event_q25": np.nan, "event_q75": np.nan,
                        "control_center": c_mean[i], "control_q25": np.nan, "control_q75": np.nan,
                        "n_event_valid": int(n_event_valid[i]), "n_control_valid": int(n_control_valid[i]),
                    })

            ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
            ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{SITE_LABELS[site]}\nn = {len(events)} events", fontsize=11)
            if row == len(PLOT_VARIABLES) - 1:
                ax.set_xlabel("Time from observed STD(t) peak (h)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "Meteorological conditions associated with observation-defined rapid thermal-structure weakening events",
        fontsize=13,
        y=0.972,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / "Figure9_meteorological_composites"
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(output_rows).to_csv(outdir / "Figure9_composite_values.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild JoHRS Figure 9 from final event catalogs.")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--before-hours", type=float, default=6.0)
    p.add_argument("--after-hours", type=float, default=6.0)
    p.add_argument("--controls-per-event", type=int, default=5)
    p.add_argument("--control-exclusion-hours", type=float, default=12.0)
    p.add_argument("--min-valid-fraction", type=float, default=0.80)
    p.add_argument("--dpi", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else root / "revision_figures" / "Figure9_meteorological_composites_v1"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    yama_events_path = find_one(root, "Shin_Yamanokami_R1_5_events_return.zip")
    shiode_events_path = find_one(root, "Shiode_R1_5_step1B_return.zip")
    yama_events, yama_obs = load_yamanokami_catalog(yama_events_path)
    shiode_events, shiode_obs = load_shiode_catalog(shiode_events_path)
    require(len(yama_events) == 13, f"Expected 13 Yamanokami events, found {len(yama_events)}")
    require(len(shiode_events) == 91, f"Expected 91 Shiode events, found {len(shiode_events)}")

    yama_met_raw, yama_met_source = read_csv_direct_or_tar(
        root, "yamanokami_2024_aligned.csv", "GLM_yamanokami_audit_20261024_130002_234593.tar.gz"
    )
    shiode_met_raw, shiode_met_source = read_csv_direct_or_tar(
        root, "shiode_2025_aligned.csv", "GLM_shiode_audit_20261022_142810_892189.tar.gz"
    )
    yama_met = clean_meteorology(yama_met_raw, yama_met_source)
    shiode_met = clean_meteorology(shiode_met_raw, shiode_met_source)

    site_results = {
        "shiode": prepare_site(
            "shiode", shiode_events, shiode_obs, shiode_met,
            args.before_hours, args.after_hours, args.controls_per_event,
            args.control_exclusion_hours, args.min_valid_fraction,
        ),
        "yamanokami": prepare_site(
            "yamanokami", yama_events, yama_obs, yama_met,
            args.before_hours, args.after_hours, args.controls_per_event,
            args.control_exclusion_hours, args.min_valid_fraction,
        ),
    }

    all_events = pd.concat([r["events_used"] for r in site_results.values()], ignore_index=True)
    all_controls = pd.concat([r["controls"] for r in site_results.values()], ignore_index=True)
    all_events.to_csv(outdir / "Figure9_events_used.csv", index=False, encoding="utf-8-sig")
    all_controls.to_csv(outdir / "Figure9_matched_controls.csv", index=False, encoding="utf-8-sig")

    audit_rows = []
    for site, result in site_results.items():
        audit_rows.append({
            "site": site,
            "final_catalog_events": len(result["events_all"]),
            "events_used_in_composite": len(result["events_used"]),
            "events_excluded_by_composite_window": len(result["events_all"]) - len(result["events_used"]),
            "matched_control_windows": len(result["controls"]),
            "controls_per_retained_event_mean": len(result["controls"]) / len(result["events_used"]),
            "time_step_minutes": result["dt_minutes"],
        })
    pd.DataFrame(audit_rows).to_csv(outdir / "Figure9_audit_summary.csv", index=False, encoding="utf-8-sig")

    provenance = {
        "event_catalogs": {
            "Yamanokami": str(yama_events_path),
            "Shiode": str(shiode_events_path),
        },
        "meteorology_sources": {
            "Yamanokami": yama_met_source,
            "Shiode": shiode_met_source,
        },
        "final_catalog_counts": {"Yamanokami": 13, "Shiode": 91, "Shin": 0},
        "alignment": "observed STD(t) peak at t=0",
        "window_hours": [-args.before_hours, args.after_hours],
        "controls_per_event": args.controls_per_event,
        "control_matching": "same month and same local time; fallback to same local time; temporally nearest",
        "control_exclusion_hours_from_any_final_event_peak": args.control_exclusion_hours,
        "minimum_valid_fraction_per_composite_window": args.min_valid_fraction,
        "continuous_summary": "median and interquartile range",
        "precipitation_summary": "mean amount in source units (aligned files contain mm per 10-min interval)",
        "causal_interpretation": "association only; not direct mechanism attribution",
    }
    (outdir / "Figure9_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    plot_and_export(site_results, outdir, args.dpi)

    print("\nFigure 9 completed")
    print("Output directory:", outdir)
    for row in audit_rows:
        print(
            f"{row['site']}: final={row['final_catalog_events']}, "
            f"used={row['events_used_in_composite']}, controls={row['matched_control_windows']}"
        )


if __name__ == "__main__":
    main()
