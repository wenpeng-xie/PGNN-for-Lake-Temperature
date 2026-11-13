#!/usr/bin/env python3
"""Rebuild Figure 8 event/member drop-ratio data from the verified archives.

Default definition follows the revised Supplementary Table C1:
modeled drop = maximum ordered decline inside the same observation-defined 3-h
window, i.e. max(STD[i] - STD[j]) for i < j. A model with no decline receives
zero drop rather than an unordered max-min range.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import figure8_loaders as loaders


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_maximum_decline(values: np.ndarray) -> Tuple[float, int | None, int | None]:
    """Return maximum ordered decline max(x[i]-x[j]) for i<j, clipped at 0."""
    x = np.asarray(values, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(x))
    if valid_idx.size < 2:
        return np.nan, None, None

    best_drop = 0.0
    best_i = int(valid_idx[0])
    best_j = int(valid_idx[1])
    running_max = float(x[valid_idx[0]])
    running_max_idx = int(valid_idx[0])

    for idx in valid_idx[1:]:
        idx = int(idx)
        drop = running_max - float(x[idx])
        if drop > best_drop:
            best_drop = drop
            best_i = running_max_idx
            best_j = idx
        if float(x[idx]) > running_max:
            running_max = float(x[idx])
            running_max_idx = idx

    return float(max(best_drop, 0.0)), best_i, best_j


def unordered_range(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if np.isfinite(x).sum() < 2:
        return np.nan
    return float(np.nanmax(x) - np.nanmin(x))


def locate_time(time: pd.DatetimeIndex, t: pd.Timestamp, label: str) -> int:
    idx = time.get_indexer([pd.Timestamp(t)])[0]
    if idx < 0:
        raise ValueError(f"Could not locate {label}={t} on the 10-min reference grid")
    return int(idx)


def build_member_table(
    site_label: str,
    event_df: pd.DataFrame,
    ref_time: pd.DatetimeIndex,
    model_series: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for _, ev in event_df.iterrows():
        event_id = str(ev["event_id"])
        start_time = pd.Timestamp(ev["start_time"])
        peak_time = pd.Timestamp(ev["peak_time"])
        min_time = pd.Timestamp(ev["min_time"])
        end_time = pd.Timestamp(ev["end_time"])
        obs_drop = float(ev["drop_degC"])
        if not np.isfinite(obs_drop) or obs_drop <= 0:
            raise ValueError(f"{site_label} {event_id}: invalid observed drop {obs_drop}")

        i0 = locate_time(ref_time, start_time, "start_time")
        i1 = locate_time(ref_time, end_time, "end_time")
        if i1 <= i0:
            raise ValueError(f"{site_label} {event_id}: invalid event indices {i0}, {i1}")

        event_time = ref_time[i0:i1 + 1]
        for model in loaders.MODEL_ORDER:
            members = model_series[model]
            if len(members) != 10:
                raise ValueError(f"{site_label} {model}: expected 10 members, got {len(members)}")
            for member, series in members.items():
                segment = np.asarray(series[i0:i1 + 1], dtype=float)
                ordered_drop, rel_peak, rel_min = ordered_maximum_decline(segment)
                range_drop = unordered_range(segment)
                finite_count = int(np.isfinite(segment).sum())
                ordered_peak_time = pd.NaT if rel_peak is None else event_time[rel_peak]
                ordered_min_time = pd.NaT if rel_min is None else event_time[rel_min]
                rows.append({
                    "site": site_label,
                    "event_id": event_id,
                    "start_time": start_time,
                    "obs_peak_time": peak_time,
                    "obs_min_time": min_time,
                    "end_time": end_time,
                    "obs_drop_degC": obs_drop,
                    "model": model,
                    "member": member,
                    "n_window_samples": len(segment),
                    "n_finite_model_samples": finite_count,
                    "ordered_model_drop_degC": ordered_drop,
                    "ordered_drop_ratio": ordered_drop / obs_drop if np.isfinite(ordered_drop) else np.nan,
                    "ordered_model_peak_time": ordered_peak_time,
                    "ordered_model_min_time": ordered_min_time,
                    "unordered_model_range_degC": range_drop,
                    "unordered_range_ratio": range_drop / obs_drop if np.isfinite(range_drop) else np.nan,
                })
    return pd.DataFrame(rows)


def aggregate_event_table(member_df: pd.DataFrame) -> pd.DataFrame:
    grouped = member_df.groupby(["site", "event_id", "model"], sort=False)
    out = grouped.agg(
        start_time=("start_time", "first"),
        obs_peak_time=("obs_peak_time", "first"),
        obs_min_time=("obs_min_time", "first"),
        end_time=("end_time", "first"),
        obs_drop_degC=("obs_drop_degC", "first"),
        n_members=("member", "nunique"),
        ordered_drop_ratio_mean=("ordered_drop_ratio", "mean"),
        ordered_drop_ratio_sd=("ordered_drop_ratio", "std"),
        ordered_drop_ratio_median=("ordered_drop_ratio", "median"),
        unordered_range_ratio_mean=("unordered_range_ratio", "mean"),
        unordered_range_ratio_sd=("unordered_range_ratio", "std"),
        unordered_range_ratio_median=("unordered_range_ratio", "median"),
    ).reset_index()
    if not (out["n_members"] == 10).all():
        bad = out[out["n_members"] != 10]
        raise ValueError(f"Some event/model combinations do not have 10 members:\n{bad}")
    return out


def aggregate_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    return (
        event_df.groupby(["site", "model"], sort=False)
        .agg(
            n_events=("event_id", "nunique"),
            event_mean_drop_ratio_mean=("ordered_drop_ratio_mean", "mean"),
            event_mean_drop_ratio_sd=("ordered_drop_ratio_mean", "std"),
            event_mean_drop_ratio_median=("ordered_drop_ratio_mean", "median"),
            event_mean_drop_ratio_q25=("ordered_drop_ratio_mean", lambda x: x.quantile(0.25)),
            event_mean_drop_ratio_q75=("ordered_drop_ratio_mean", lambda x: x.quantile(0.75)),
            event_mean_drop_ratio_min=("ordered_drop_ratio_mean", "min"),
            event_mean_drop_ratio_max=("ordered_drop_ratio_mean", "max"),
        )
        .reset_index()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Directory containing the seven verified archives.")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory; defaults to the script directory/data.")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve() if args.outdir else Path(__file__).resolve().parent / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "yama_events": loaders.find_file(root, "Shin_Yamanokami_R1_5_events_return.zip"),
        "shiode_events": loaders.find_file(root, "Shiode_R1_5_step1B_return.zip"),
        "yama_glm": loaders.find_file(root, "GLM_yamanokami_return.tar.gz"),
        "shiode_glm": loaders.find_file(root, "GLM_shiode_test_return.tar.gz"),
        "yama_ml": loaders.find_file(root, "Yamanokami_ML_original_batch_20261024_170442_830097.zip"),
        "yama_pgnn_added": loaders.find_file(root, "Yamanokami_PGNN_added_seeds_20261024_173009.zip"),
        "shiode_ml": loaders.find_file(root, "Shiode_ML_final_members_20261025_181111_032810.zip"),
    }

    yama_events, yama_obs = loaders.load_yamanokami_event_data(inputs["yama_events"])
    shiode_events, shiode_obs = loaders.load_shiode_event_data(inputs["shiode_events"])
    if len(yama_events) != 13 or len(shiode_events) != 91:
        raise ValueError(f"Unexpected final event counts: Yamanokami={len(yama_events)}, Shiode={len(shiode_events)}")

    yama_time = pd.DatetimeIndex(yama_obs["time"])
    shiode_time = pd.DatetimeIndex(shiode_obs["time"])

    yama_models: Dict[str, Dict[str, np.ndarray]] = {
        "GLM": loaders.load_glm_members_yamanokami(inputs["yama_glm"], yama_time)
    }
    yama_models.update(loaders.load_yamanokami_ml_members(
        inputs["yama_ml"], inputs["yama_pgnn_added"], yama_time
    ))

    shiode_models: Dict[str, Dict[str, np.ndarray]] = {
        "GLM": loaders.load_glm_members_shiode(inputs["shiode_glm"], shiode_time)
    }
    shiode_models.update(loaders.load_shiode_ml_members(inputs["shiode_ml"], shiode_time))

    yama_member = build_member_table("Yamanokami", yama_events, yama_time, yama_models)
    shiode_member = build_member_table("Shiode", shiode_events, shiode_time, shiode_models)
    member_df = pd.concat([yama_member, shiode_member], ignore_index=True)
    event_df = aggregate_event_table(member_df)
    summary_df = aggregate_summary(event_df)

    member_path = outdir / "Figure8_member_level_drop_ratios.csv"
    event_path = outdir / "Figure8_event_level_drop_ratios.csv"
    summary_path = outdir / "Figure8_summary_by_site_model.csv"
    member_df.to_csv(member_path, index=False, encoding="utf-8-sig")
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    provenance = {
        "definition_used_in_main_plot": (
            "For each member and observation-defined 3-h window, ordered modeled drop is "
            "max(STD[i]-STD[j]) over i<j, clipped at zero; event point is the arithmetic mean "
            "of the 10 member-level ratios."
        ),
        "legacy_audit_quantity": "Unordered max(STD)-min(STD) is retained in the CSV but is not plotted by default.",
        "event_counts": {"Yamanokami": 13, "Shiode": 91, "Shin": 0},
        "fixed_depths_m": {
            "Yamanokami": loaders.YAMA_DEPTHS,
            "Shiode": loaders.SHIODE_DEPTHS,
        },
        "input_files": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in inputs.items()
        },
        "member_policies": {
            "Yamanokami": loaders.YAMA_MEMBER_POLICY,
            "Shiode": loaders.SHIODE_MEMBER_POLICY,
        },
    }
    (outdir / "Figure8_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("Saved:")
    print(member_path)
    print(event_path)
    print(summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
