#!/usr/bin/env python3
"""Rebuild Supplementary Figure C2 with the final event/drop definition.

Final definition:
- events come from the fixed-depth, complete-window threshold catalogs;
- for every frozen member, modeled drop is the maximum ordered decline
  max(STD[i]-STD[j]) for i<j inside the same observation-defined 3-h window;
- one plotted event value is the arithmetic mean of the 10 member-level ratios;
- boxes summarize events, not event-member pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import figureC2_loaders as loaders

THRESHOLD_ORDER = ["-50%", "-25%", "Original", "+25%", "+50%"]
SITE_ORDER = ["Shiode", "Yamanokami"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_maximum_decline(values: np.ndarray) -> Tuple[float, int | None, int | None]:
    x = np.asarray(values, dtype=float)
    idxs = np.flatnonzero(np.isfinite(x))
    if idxs.size < 2:
        return np.nan, None, None
    running_max = float(x[idxs[0]])
    running_idx = int(idxs[0])
    best = 0.0
    best_i = running_idx
    best_j = int(idxs[1])
    for idx in idxs[1:]:
        idx = int(idx)
        drop = running_max - float(x[idx])
        if drop > best:
            best = drop
            best_i = running_idx
            best_j = idx
        if float(x[idx]) > running_max:
            running_max = float(x[idx])
            running_idx = idx
    return float(max(best, 0.0)), best_i, best_j


def locate_time(time: pd.DatetimeIndex, t: pd.Timestamp, label: str) -> int:
    pos = int(time.get_indexer([pd.Timestamp(t)])[0])
    if pos < 0:
        raise ValueError(f"Could not locate {label}={t} on reference grid")
    return pos


def load_threshold_catalog(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        df = pd.read_csv(io.BytesIO(zf.read("threshold_event_catalogs.csv")))
    for c in ["start_time", "peak_time", "min_time", "end_time"]:
        df[c] = pd.to_datetime(df[c])
    df = df[df["site"].isin(SITE_ORDER)].copy()
    df["site"] = pd.Categorical(df["site"], SITE_ORDER, ordered=True)
    df["threshold_setting"] = pd.Categorical(df["threshold_setting"], THRESHOLD_ORDER, ordered=True)
    return df.sort_values(["site", "threshold_setting", "event_id"]).reset_index(drop=True)


def build_member_table(
    catalog: pd.DataFrame,
    site: str,
    ref_time: pd.DatetimeIndex,
    model_series: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    sdf = catalog[catalog["site"].astype(str) == site].copy()
    for _, ev in sdf.iterrows():
        obs_drop = float(ev["obs_amplitude_degC"])
        if not np.isfinite(obs_drop) or obs_drop <= 0:
            raise ValueError(f"{site} {ev['threshold_setting']} event {ev['event_id']}: bad obs amplitude")
        i0 = locate_time(ref_time, ev["start_time"], "start_time")
        i1 = locate_time(ref_time, ev["end_time"], "end_time")
        if i1 <= i0:
            raise ValueError(f"{site}: invalid event interval")
        event_time = ref_time[i0:i1+1]
        for model in loaders.MODEL_ORDER:
            members = model_series[model]
            if len(members) != 10:
                raise ValueError(f"{site} {model}: expected 10 members, got {len(members)}")
            for member, series in members.items():
                seg = np.asarray(series[i0:i1+1], dtype=float)
                drop, pi, mi = ordered_maximum_decline(seg)
                rows.append({
                    "site": site,
                    "threshold_setting": str(ev["threshold_setting"]),
                    "event_id": int(ev["event_id"]),
                    "start_time": pd.Timestamp(ev["start_time"]),
                    "obs_peak_time": pd.Timestamp(ev["peak_time"]),
                    "obs_min_time": pd.Timestamp(ev["min_time"]),
                    "end_time": pd.Timestamp(ev["end_time"]),
                    "obs_drop_degC": obs_drop,
                    "model": model,
                    "member": member,
                    "n_window_samples": len(seg),
                    "n_finite_model_samples": int(np.isfinite(seg).sum()),
                    "ordered_model_drop_degC": drop,
                    "drop_ratio": drop / obs_drop if np.isfinite(drop) else np.nan,
                    "model_peak_time": pd.NaT if pi is None else event_time[pi],
                    "model_min_time": pd.NaT if mi is None else event_time[mi],
                })
    return pd.DataFrame(rows)


def aggregate_event_table(member_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        member_df.groupby(["site", "threshold_setting", "event_id", "model"], sort=False)
        .agg(
            start_time=("start_time", "first"),
            obs_peak_time=("obs_peak_time", "first"),
            obs_min_time=("obs_min_time", "first"),
            end_time=("end_time", "first"),
            obs_drop_degC=("obs_drop_degC", "first"),
            n_members=("member", "nunique"),
            drop_ratio_mean=("drop_ratio", "mean"),
            drop_ratio_sd=("drop_ratio", "std"),
            drop_ratio_median=("drop_ratio", "median"),
        )
        .reset_index()
    )
    bad = out[out["n_members"] != 10]
    if len(bad):
        raise ValueError(f"Event/model rows without 10 members:\n{bad}")
    return out


def aggregate_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    return (
        event_df.groupby(["site", "threshold_setting", "model"], sort=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_event_ratio=("drop_ratio_mean", "mean"),
            median_event_ratio=("drop_ratio_mean", "median"),
            q25_event_ratio=("drop_ratio_mean", lambda x: x.quantile(0.25)),
            q75_event_ratio=("drop_ratio_mean", lambda x: x.quantile(0.75)),
            median_abs_ratio_error=("drop_ratio_mean", lambda x: np.median(np.abs(x - 1.0))),
            mean_abs_ratio_error=("drop_ratio_mean", lambda x: np.mean(np.abs(x - 1.0))),
        )
        .reset_index()
    )


def build_rankings(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    consistency = []
    for site in SITE_ORDER:
        original_order = None
        for th in THRESHOLD_ORDER:
            d = summary[(summary["site"] == site) & (summary["threshold_setting"] == th)].copy()
            d = d.sort_values(["median_abs_ratio_error", "model"])
            d["rank_by_median_abs_ratio_error"] = np.arange(1, len(d)+1)
            rows.append(d)
            order = " > ".join(d["model"].tolist())
            if th == "Original":
                original_order = order
        for th in THRESHOLD_ORDER:
            d = summary[(summary["site"] == site) & (summary["threshold_setting"] == th)].copy()
            d = d.sort_values(["median_abs_ratio_error", "model"])
            order = " > ".join(d["model"].tolist())
            consistency.append({
                "site": site,
                "threshold_setting": th,
                "original_rank_order": original_order,
                "current_rank_order": order,
                "same_rank_order_as_original": order == original_order,
                "n_events": int(d["n_events"].iloc[0]),
            })
    ranking = pd.concat(rows, ignore_index=True)
    return ranking, pd.DataFrame(consistency)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve() if args.outdir else Path(__file__).resolve().parent / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "threshold_return": loaders.find_file(root, "JoHRS_R2_3_threshold_sensitivity_return.zip"),
        "yama_events": loaders.find_file(root, "Shin_Yamanokami_R1_5_events_return.zip"),
        "shiode_events": loaders.find_file(root, "Shiode_R1_5_step1B_return.zip"),
        "yama_glm": loaders.find_file(root, "GLM_yamanokami_return.tar.gz"),
        "shiode_glm": loaders.find_file(root, "GLM_shiode_test_return.tar.gz"),
        "yama_ml": loaders.find_file(root, "Yamanokami_ML_original_batch_20261024_170442_830097.zip"),
        "yama_pgnn_added": loaders.find_file(root, "Yamanokami_PGNN_added_seeds_20261024_173009.zip"),
        "shiode_ml": loaders.find_file(root, "Shiode_ML_final_members_20261025_181111_032810.zip"),
    }

    catalog = load_threshold_catalog(inputs["threshold_return"])
    yama_final, yama_obs = loaders.load_yamanokami_event_data(inputs["yama_events"])
    shiode_final, shiode_obs = loaders.load_shiode_event_data(inputs["shiode_events"])
    if len(yama_final) != 13 or len(shiode_final) != 91:
        raise ValueError("Final event catalogs are not 13/91")

    ytime = pd.DatetimeIndex(yama_obs["time"])
    stime = pd.DatetimeIndex(shiode_obs["time"])

    ymodels: Dict[str, Dict[str, np.ndarray]] = {"GLM": loaders.load_glm_members_yamanokami(inputs["yama_glm"], ytime)}
    ymodels.update(loaders.load_yamanokami_ml_members(inputs["yama_ml"], inputs["yama_pgnn_added"], ytime))
    smodels: Dict[str, Dict[str, np.ndarray]] = {"GLM": loaders.load_glm_members_shiode(inputs["shiode_glm"], stime)}
    smodels.update(loaders.load_shiode_ml_members(inputs["shiode_ml"], stime))

    member = pd.concat([
        build_member_table(catalog, "Yamanokami", ytime, ymodels),
        build_member_table(catalog, "Shiode", stime, smodels),
    ], ignore_index=True)
    event = aggregate_event_table(member)
    summary = aggregate_summary(event)
    ranking, consistency = build_rankings(summary)

    member.to_csv(outdir / "FigureC2_member_level_ratios.csv", index=False, encoding="utf-8-sig")
    event.to_csv(outdir / "FigureC2_event_level_ratios.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "FigureC2_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(outdir / "FigureC2_ranking.csv", index=False, encoding="utf-8-sig")
    consistency.to_csv(outdir / "FigureC2_rank_consistency.csv", index=False, encoding="utf-8-sig")

    provenance = {
        "final_definition": (
            "Per frozen member, modeled drop is maximum ordered decline max(STD[i]-STD[j]) for i<j "
            "within the same observation-defined 3-h window; event value is mean of 10 member ratios; "
            "boxes summarize events."
        ),
        "thresholds": THRESHOLD_ORDER,
        "event_counts": {
            f"{site}|{threshold}": int(count)
            for (site, threshold), count in catalog.groupby(
                [catalog["site"].astype(str), catalog["threshold_setting"].astype(str)]
            ).size().items()
        },
        "fixed_depths_m": {"Yamanokami": loaders.YAMA_DEPTHS, "Shiode": loaders.SHIODE_DEPTHS},
        "input_files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in inputs.items()},
        "member_policies": {"Yamanokami": loaders.YAMA_MEMBER_POLICY, "Shiode": loaders.SHIODE_MEMBER_POLICY},
    }
    (outdir / "FigureC2_provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nRank consistency:\n", consistency.to_string(index=False))


if __name__ == "__main__":
    main()
