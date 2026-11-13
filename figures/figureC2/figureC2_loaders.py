from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Defaults / policies for the current JoHRS revision
# ============================================================
YAMA_DEPTHS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
SHIODE_DEPTHS = [0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85, 2.05, 2.25]

MODEL_ORDER = ["GLM", "LSTM", "rSSM", "PGNN"]
MODEL_COLORS = {
    "Observed": "black",
    "GLM": "#666666",
    "LSTM": "#1f77b4",
    "rSSM": "#ff7f0e",
    "PGNN": "#2ca02c",
}

SITE_TITLES = {
    "yamanokami": "Yamanokami",
    "shiode": "Shiode",
}

# final member policies already verified earlier in this project
YAMA_MEMBER_POLICY = {
    "LSTM": ["base"] + [f"seed_{i}" for i in range(1, 10)],
    "rSSM": ["base"] + [f"seed_{i}" for i in range(1, 10)],
    "PGNN": ["seed3", "seed4", "seed5", "seed6", "seed7", "seed9", "seed14", "seed15", "seed16", "seed20"],
}
SHIODE_MEMBER_POLICY = {
    "LSTM": ["base"] + [f"seed{i}" for i in range(1, 10)],
    "rSSM": ["base"] + [f"seed{i}" for i in range(1, 10)],
    "PGNN": ["seed0", "seed1", "seed3", "seed5", "seed6", "seed7", "seed9", "seed41", "seed100", "seed256"],
}


@dataclass
class EventRecord:
    site: str
    rank_within_site: int
    event_id: str
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    min_time: pd.Timestamp
    end_time: pd.Timestamp
    drop_degC: float


# ============================================================
# utilities
# ============================================================
def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def find_file(root: Path, preferred_name: str) -> Path:
    direct = root / preferred_name
    if direct.exists():
        return direct
    matches = list(root.rglob(preferred_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches = sorted(matches)
        return matches[0]
    raise FileNotFoundError(f"Could not find required input: {preferred_name} under {root}")


def to_timestamp_array(arr: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(arr.astype("datetime64[ns]"))


def nearest_depth_indices(depths: Sequence[float], target_depths: Sequence[float], tol: float = 0.12) -> List[int]:
    depths = np.asarray(depths, dtype=float)
    idxs: List[int] = []
    for td in target_depths:
        diffs = np.abs(depths - td)
        i = int(np.argmin(diffs))
        if diffs[i] > tol:
            raise ValueError(
                f"Could not match target depth {td} m within tolerance {tol}; available depths={depths.tolist()}"
            )
        idxs.append(i)
    return idxs


def calc_std_series(temp: np.ndarray) -> np.ndarray:
    temp = np.asarray(temp, dtype=float)
    out = np.full(temp.shape[0], np.nan, dtype=float)
    finite = np.isfinite(temp)
    n = finite.sum(axis=1)
    good = n >= 2
    if np.any(good):
        with np.errstate(all="ignore"):
            out[good] = np.nanstd(temp[good, :], axis=1, ddof=0)
    return out


def align_series_to_reference(
    ref_time: pd.DatetimeIndex,
    src_time: pd.DatetimeIndex,
    src_values: np.ndarray,
) -> np.ndarray:
    src_df = pd.DataFrame({"time": src_time, "value": src_values})
    ref_df = pd.DataFrame({"time": ref_time})
    merged = ref_df.merge(src_df, on="time", how="left")
    return merged["value"].to_numpy(dtype=float)


def common_member_stats(series_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(series_list) > 0, "series_list is empty")
    stack = np.vstack(series_list)
    valid = np.isfinite(stack)
    mean = np.full(stack.shape[1], np.nan, dtype=float)
    sd = np.full(stack.shape[1], np.nan, dtype=float)
    count = valid.sum(axis=0)
    good1 = count >= 1
    if np.any(good1):
        mean[good1] = np.nanmean(stack[:, good1], axis=0)
    good2 = count >= 2
    if np.any(good2):
        sd[good2] = np.nanstd(stack[:, good2], axis=0, ddof=1)
    sd[np.isnan(sd)] = 0.0
    return mean, sd, count


# ============================================================
# event catalogs and observed structure series
# ============================================================
def load_yamanokami_event_data(event_zip: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(event_zip) as zf:
        ev = pd.read_csv(io.BytesIO(zf.read("yamanokami/core10/events_B.csv")))
        obs = pd.read_csv(io.BytesIO(zf.read("yamanokami/core10/observation_series.csv.gz")), compression="gzip")
    for c in ["start_time", "peak_time", "min_time", "end_time"]:
        ev[c] = pd.to_datetime(ev[c])
    obs["time"] = pd.to_datetime(obs["time"])
    return ev, obs


def load_shiode_event_data(step1b_zip: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(step1b_zip) as zf:
        ev = pd.read_csv(io.BytesIO(zf.read("input_events_B_fixed11_complete3h.csv")))
        npz = np.load(io.BytesIO(zf.read("ensemble_structure_series.npz")))
        obs = pd.DataFrame({
            "time": to_timestamp_array(npz["time"]),
            "STD_B_complete": npz["obs_STD_fixed11"],
        })
    for c in ["start_time", "peak_time", "min_time", "end_time"]:
        ev[c] = pd.to_datetime(ev[c])
    return ev, obs


def select_top_events(ev: pd.DataFrame, site: str, top_n: int = 3) -> List[EventRecord]:
    df = ev.copy()
    require("drop_degC" in df.columns, f"{site}: event table missing drop_degC")
    df = df.sort_values(["drop_degC", "start_time"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    out: List[EventRecord] = []
    for i, row in df.iterrows():
        out.append(
            EventRecord(
                site=site,
                rank_within_site=i + 1,
                event_id=str(row["event_id"]),
                start_time=pd.Timestamp(row["start_time"]),
                peak_time=pd.Timestamp(row["peak_time"]),
                min_time=pd.Timestamp(row["min_time"]),
                end_time=pd.Timestamp(row["end_time"]),
                drop_degC=float(row["drop_degC"]),
            )
        )
    return out


# ============================================================
# model loaders
# ============================================================
def load_glm_members_yamanokami(glm_tar: Path, ref_time: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with tarfile.open(glm_tar, "r:gz") as tf:
        names = [n for n in tf.getnames() if "/predictions/" in n and n.endswith(".npz")]
        require(len(names) == 10, f"Expected 10 Yamanokami GLM members, found {len(names)}")
        for name in sorted(names):
            member = Path(name).stem
            npz = np.load(io.BytesIO(tf.extractfile(name).read()))
            time = to_timestamp_array(npz["time"])
            depths = npz["depths"].astype(float)
            pred = np.asarray(npz["pred"], dtype=float)
            idxs = nearest_depth_indices(depths, YAMA_DEPTHS, tol=0.11)
            std = calc_std_series(pred[:, idxs])
            out[member] = align_series_to_reference(ref_time, time, std)
    return out


def load_glm_members_shiode(glm_tar: Path, ref_time: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with tarfile.open(glm_tar, "r:gz") as tf:
        names = [n for n in tf.getnames() if n.startswith("predictions/") and n.endswith(".npz")]
        require(len(names) == 10, f"Expected 10 Shiode GLM members, found {len(names)}")
        for name in sorted(names):
            member = Path(name).stem
            npz = np.load(io.BytesIO(tf.extractfile(name).read()))
            time = to_timestamp_array(npz["time"])
            depths = npz["depths"].astype(float)
            pred = np.asarray(npz["pred"], dtype=float)
            idxs = nearest_depth_indices(depths, SHIODE_DEPTHS, tol=0.11)
            std = calc_std_series(pred[:, idxs])
            out[member] = align_series_to_reference(ref_time, time, std)
    return out


def _load_npz_member_std_from_zip(
    zf: zipfile.ZipFile,
    npz_path: str,
    target_obs_depths: Sequence[float],
    ref_time: pd.DatetimeIndex,
) -> np.ndarray:
    raw = io.BytesIO(zf.read(npz_path))
    npz = np.load(raw)
    time = to_timestamp_array(npz["time"])
    pred = np.asarray(npz["pred_sel"], dtype=float)
    matched_obs_depths = np.asarray(npz["matched_obs_depths"], dtype=float)
    warmup_len = int(np.asarray(npz["warmup_len"]).item())
    idxs = nearest_depth_indices(matched_obs_depths, target_obs_depths, tol=0.11)
    std = calc_std_series(pred[:, idxs])
    std[:warmup_len] = np.nan
    return align_series_to_reference(ref_time, time, std)


def _load_csv_member_std_from_zip(
    zf: zipfile.ZipFile,
    csv_path: str,
    target_depths: Sequence[float],
    ref_time: pd.DatetimeIndex,
) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(zf.read(csv_path)))
    df["time"] = pd.to_datetime(df["time"])
    depth_cols = [c for c in df.columns if c.endswith("m")]
    parsed = np.array([float(c[:-1]) for c in depth_cols], dtype=float)
    idxs = nearest_depth_indices(parsed, target_depths, tol=0.11)
    chosen_cols = [depth_cols[i] for i in idxs]
    temp = df[chosen_cols].to_numpy(dtype=float)
    std = calc_std_series(temp)
    return align_series_to_reference(ref_time, pd.DatetimeIndex(df["time"]), std)


def load_yamanokami_ml_members(ml_zip: Path, pgnn_added_zip: Path, ref_time: pd.DatetimeIndex) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in ["LSTM", "rSSM", "PGNN"]}
    with zipfile.ZipFile(ml_zip) as zf:
        # LSTM
        out["LSTM"]["base"] = _load_npz_member_std_from_zip(
            zf,
            "evidence/runs_qc/yamanokami_2024_lstm_overlap_eval_base/lstm_overlap_avg_predictions.npz",
            YAMA_DEPTHS,
            ref_time,
        )
        for i in range(1, 10):
            out["LSTM"][f"seed_{i}"] = _load_npz_member_std_from_zip(
                zf,
                f"evidence/runs_qc/yamanokami_2024_lstm_overlap_eval_base/seed_{i}/lstm_overlap_avg_predictions.npz",
                YAMA_DEPTHS,
                ref_time,
            )
        # rSSM
        out["rSSM"]["base"] = _load_npz_member_std_from_zip(
            zf,
            "evidence/runs_qc/yamanokami_2024_ssm_overlap_eval_base/ssm_overlap_avg_predictions.npz",
            YAMA_DEPTHS,
            ref_time,
        )
        for i in range(1, 10):
            out["rSSM"][f"seed_{i}"] = _load_npz_member_std_from_zip(
                zf,
                f"evidence/runs_qc/yamanokami_2024_ssm_overlap_eval_structure/seed_{i}/ssm_overlap_avg_predictions.npz",
                YAMA_DEPTHS,
                ref_time,
            )
        # original PGNN members that remain in the final set
        for seed in [3, 4, 5, 6, 7, 9]:
            out["PGNN"][f"seed{seed}"] = _load_npz_member_std_from_zip(
                zf,
                f"evidence/runs_qc/yamanokami_2024_lstm_overlap_eval_structure_sgh/seed{seed}/lstm_overlap_avg_predictions.npz",
                YAMA_DEPTHS,
                ref_time,
            )
    with zipfile.ZipFile(pgnn_added_zip) as zf:
        for seed in [14, 15, 16, 20]:
            matched = [n for n in zf.namelist() if n.replace("\\", "/") == f"seed{seed}/lstm_overlap_avg_predictions.npz"]
            require(len(matched) == 1, f"Could not uniquely find Yamanokami added PGNN seed{seed}")
            out["PGNN"][f"seed{seed}"] = _load_npz_member_std_from_zip(zf, matched[0], YAMA_DEPTHS, ref_time)
    # enforce exact final member set
    for model, wanted in YAMA_MEMBER_POLICY.items():
        missing = [m for m in wanted if m not in out[model]]
        require(len(missing) == 0, f"Yamanokami {model}: missing members {missing}")
        out[model] = {m: out[model][m] for m in wanted}
    return out


def load_shiode_ml_members(ml_zip: Path, ref_time: pd.DatetimeIndex) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in ["LSTM", "rSSM", "PGNN"]}
    with zipfile.ZipFile(ml_zip) as zf:
        # LSTM
        out["LSTM"]["base"] = _load_npz_member_std_from_zip(
            zf,
            "evidence/runs_qc/shiode_2025_lstm_overlap_eval_base/lstm_overlap_avg_predictions.npz",
            SHIODE_DEPTHS,
            ref_time,
        )
        for i in range(1, 10):
            out["LSTM"][f"seed{i}"] = _load_npz_member_std_from_zip(
                zf,
                f"evidence/runs_qc/shiode_2025_lstm_overlap_eval_base/seed_{i}/lstm_overlap_avg_predictions.npz",
                SHIODE_DEPTHS,
                ref_time,
            )
        # rSSM
        out["rSSM"]["base"] = _load_npz_member_std_from_zip(
            zf,
            "evidence/runs_qc/shiode_2025_ssm_overlap_eval_base/ssm_overlap_avg_predictions.npz",
            SHIODE_DEPTHS,
            ref_time,
        )
        for i in range(1, 10):
            out["rSSM"][f"seed{i}"] = _load_npz_member_std_from_zip(
                zf,
                f"evidence/runs_qc/shiode_2025_ssm_overlap_eval_structure/seed_{i}/ssm_overlap_avg_predictions.npz",
                SHIODE_DEPTHS,
                ref_time,
            )
        # PGNN - npz members
        pgnn_npz_paths = {
            "seed0": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed0/lstm_overlap_avg_predictions.npz",
            "seed1": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed1/lstm_overlap_avg_predictions.npz",
            "seed3": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed3/lstm_overlap_avg_predictions.npz",
            "seed5": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed5/lstm_overlap_avg_predictions.npz",
            "seed6": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed6/lstm_overlap_avg_predictions.npz",
            "seed7": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed7/lstm_overlap_avg_predictions.npz",
            "seed9": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh/seed9/lstm_overlap_avg_predictions.npz",
            "seed256": "evidence/runs_qc/shiode_2025_lstm_overlap_eval_structure_sgh_1/seed256/lstm_overlap_avg_predictions.npz",
        }
        for member, path in pgnn_npz_paths.items():
            out["PGNN"][member] = _load_npz_member_std_from_zip(zf, path, SHIODE_DEPTHS, ref_time)
        # PGNN - CSV-only members
        out["PGNN"]["seed41"] = _load_csv_member_std_from_zip(
            zf,
            "evidence/runs_qc/shiode_2025_lstm_structure_sgh_csv/shiode_lstm_structure_sgh_seed41.csv",
            SHIODE_DEPTHS,
            ref_time,
        )
        out["PGNN"]["seed100"] = _load_csv_member_std_from_zip(
            zf,
            "evidence/runs_qc/shiode_2025_lstm_structure_sgh_csv/shiode_lstm_structure_sgh_seed100.csv",
            SHIODE_DEPTHS,
            ref_time,
        )
    for model, wanted in SHIODE_MEMBER_POLICY.items():
        missing = [m for m in wanted if m not in out[model]]
        require(len(missing) == 0, f"Shiode {model}: missing members {missing}")
        out[model] = {m: out[model][m] for m in wanted}
    return out


# ============================================================
# plotting
# ============================================================
def make_selected_events_csv(events: List[EventRecord], out_csv: Path) -> None:
    df = pd.DataFrame([
        {
            "site": e.site,
            "rank_within_site": e.rank_within_site,
            "event_id": e.event_id,
            "start_time": e.start_time,
            "peak_time": e.peak_time,
            "min_time": e.min_time,
            "end_time": e.end_time,
            "drop_degC": e.drop_degC,
        }
        for e in events
    ])
    df.to_csv(out_csv, index=False)


def plot_figure(
    out_dir: Path,
    selected_events: List[EventRecord],
    obs_series_by_site: Dict[str, pd.DataFrame],
    model_series_by_site: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    hours_before: float = 24.0,
    hours_after: float = 24.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), sharex=True, sharey=False)
    letters = list("abcdef")

    # common event order: top row yamanokami, bottom row shiode
    panel_events = [e for e in selected_events if e.site == "yamanokami"] + [e for e in selected_events if e.site == "shiode"]
    require(len(panel_events) == 6, f"Expected 6 selected events, got {len(panel_events)}")

    for k, event in enumerate(panel_events):
        ax = axes[k // 3, k % 3]
        obs_df = obs_series_by_site[event.site]
        time = pd.DatetimeIndex(obs_df["time"])
        obs_std = obs_df["STD_B_complete"].to_numpy(dtype=float)

        rel_hours = (time - event.min_time) / pd.Timedelta(hours=1)
        mask = (rel_hours >= -hours_before) & (rel_hours <= hours_after)
        x = rel_hours[mask].to_numpy(dtype=float)
        y_obs = obs_std[mask]

        ax.axvspan(
            (event.start_time - event.min_time) / pd.Timedelta(hours=1),
            (event.end_time - event.min_time) / pd.Timedelta(hours=1),
            color="#dddddd",
            alpha=0.5,
            zorder=0,
        )
        ax.axvline((event.peak_time - event.min_time) / pd.Timedelta(hours=1), color="black", linestyle=":", linewidth=1.2)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)

        ax.plot(x, y_obs, color=MODEL_COLORS["Observed"], linewidth=2.2, label="Observed", zorder=5)

        site_models = model_series_by_site[event.site]
        for model in MODEL_ORDER:
            member_dict = site_models[model]
            ordered_members = list(member_dict.keys())
            member_window = [member_dict[m][mask] for m in ordered_members]
            mean, sd, _ = common_member_stats(member_window)
            ax.fill_between(x, mean - sd, mean + sd, color=MODEL_COLORS[model], alpha=0.16, linewidth=0)
            ax.plot(x, mean, color=MODEL_COLORS[model], linewidth=1.8, label=model, zorder=4)

        title = (
            f"({letters[k]}) {SITE_TITLES[event.site]} | rank {event.rank_within_site}\n"
            f"peak→min drop = {event.drop_degC:.2f} °C | min at {event.min_time:%Y-%m-%d %H:%M}"
        )
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xlim(-hours_before, hours_after)
        ax.set_xlabel("Hours relative to observed minimum STD(t)")
        if k % 3 == 0:
            ax.set_ylabel("STD(t) (°C)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    fig.legend([x[0] for x in uniq], [x[1] for x in uniq], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(
        "Figure 7. Conditional responses during observation-defined rapid thermal-structure weakening events\n"
        "Three largest observed STD-drop events at Yamanokami and Shiode; colored lines show member mean and shaded bands show ±1 sample SD.",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    png = out_dir / "Figure7_event_responses.png"
    pdf = out_dir / "Figure7_event_responses.pdf"
    svg = out_dir / "Figure7_event_responses.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def write_notes(out_dir: Path, selected_events: List[EventRecord]) -> None:
    lines = []
    lines.append("Figure 7 notes")
    lines.append("=================")
    lines.append("")
    lines.append("Selection rule:")
    lines.append("- Yamanokami: top 3 events by observed drop_degC from events_B.csv in core10.")
    lines.append("- Shiode: top 3 events by observed drop_degC from input_events_B_fixed11_complete3h.csv.")
    lines.append("- No model performance was used in the selection.")
    lines.append("")
    lines.append("Model aggregation rule:")
    lines.append("- For every member, compute STD(t) first from the fixed-depth temperatures.")
    lines.append("- Then compute the member mean and sample SD across the 10 members.")
    lines.append("")
    lines.append("Selected events:")
    for e in selected_events:
        lines.append(
            f"- {e.site} rank {e.rank_within_site}: {e.event_id}, drop={e.drop_degC:.4f} °C, "
            f"start={e.start_time}, peak={e.peak_time}, min={e.min_time}, end={e.end_time}"
        )
    (out_dir / "README_figure7_notes.txt").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# main
# ============================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Build revised JoHRS Figure 7 from the current event catalogs and current model members.")
    p.add_argument("--root", type=str, default=".", help="Root directory where the input archives are located.")
    p.add_argument("--yama-events", type=str, default="Shin_Yamanokami_R1_5_events_return.zip")
    p.add_argument("--shiode-step1b", type=str, default="Shiode_R1_5_step1B_return.zip")
    p.add_argument("--yama-glm", type=str, default="GLM_yamanokami_return.tar.gz")
    p.add_argument("--shiode-glm", type=str, default="GLM_shiode_test_return.tar.gz")
    p.add_argument("--yama-ml", type=str, default="Yamanokami_ML_original_batch_20261024_170442_830097.zip")
    p.add_argument("--yama-pgnn-added", type=str, default="Yamanokami_PGNN_added_seeds_20261024_173009.zip")
    p.add_argument("--shiode-ml", type=str, default="Shiode_ML_final_members_20261025_181111_032810.zip")
    p.add_argument("--outdir", type=str, default="revision_figures/Figure7_event_responses_v1")
    p.add_argument("--hours-before", type=float, default=24.0)
    p.add_argument("--hours-after", type=float, default=24.0)
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    yama_events_zip = find_file(root, args.yama_events)
    shiode_step1b_zip = find_file(root, args.shiode_step1b)
    yama_glm_tar = find_file(root, args.yama_glm)
    shiode_glm_tar = find_file(root, args.shiode_glm)
    yama_ml_zip = find_file(root, args.yama_ml)
    yama_pgnn_added_zip = find_file(root, args.yama_pgnn_added)
    shiode_ml_zip = find_file(root, args.shiode_ml)

    yama_ev, yama_obs = load_yamanokami_event_data(yama_events_zip)
    shiode_ev, shiode_obs = load_shiode_event_data(shiode_step1b_zip)

    yama_selected = select_top_events(yama_ev, "yamanokami", top_n=3)
    shiode_selected = select_top_events(shiode_ev, "shiode", top_n=3)
    selected_events = yama_selected + shiode_selected
    make_selected_events_csv(selected_events, out_dir / "Figure7_selected_events.csv")

    yama_ref_time = pd.DatetimeIndex(yama_obs["time"])
    shiode_ref_time = pd.DatetimeIndex(shiode_obs["time"])

    yama_models: Dict[str, Dict[str, np.ndarray]] = {}
    yama_models["GLM"] = load_glm_members_yamanokami(yama_glm_tar, yama_ref_time)
    yama_ml = load_yamanokami_ml_members(yama_ml_zip, yama_pgnn_added_zip, yama_ref_time)
    yama_models.update(yama_ml)

    shiode_models: Dict[str, Dict[str, np.ndarray]] = {}
    shiode_models["GLM"] = load_glm_members_shiode(shiode_glm_tar, shiode_ref_time)
    shiode_ml = load_shiode_ml_members(shiode_ml_zip, shiode_ref_time)
    shiode_models.update(shiode_ml)

    provenance = {
        "inputs": {
            "yama_events": str(yama_events_zip),
            "shiode_step1b": str(shiode_step1b_zip),
            "yama_glm": str(yama_glm_tar),
            "shiode_glm": str(shiode_glm_tar),
            "yama_ml": str(yama_ml_zip),
            "yama_pgnn_added": str(yama_pgnn_added_zip),
            "shiode_ml": str(shiode_ml_zip),
        },
        "selection_rule": "Top 3 observed drop_degC events within each site, from final fixed-depth event catalogs only.",
        "aggregation_rule": "Compute STD(t) per member first, then take member mean and sample SD.",
        "yamanokami_member_policy": YAMA_MEMBER_POLICY,
        "shiode_member_policy": SHIODE_MEMBER_POLICY,
        "hours_before": args.hours_before,
        "hours_after": args.hours_after,
    }
    (out_dir / "Figure7_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    obs_series_by_site = {
        "yamanokami": yama_obs[["time", "STD_B_complete"]].copy(),
        "shiode": shiode_obs[["time", "STD_B_complete"]].copy(),
    }
    model_series_by_site = {
        "yamanokami": yama_models,
        "shiode": shiode_models,
    }

    plot_figure(
        out_dir=out_dir,
        selected_events=selected_events,
        obs_series_by_site=obs_series_by_site,
        model_series_by_site=model_series_by_site,
        hours_before=args.hours_before,
        hours_after=args.hours_after,
    )
    write_notes(out_dir, selected_events)

    print("Saved outputs to:", out_dir)
    print("Selected events:")
    for e in selected_events:
        print(f"  {e.site} rank {e.rank_within_site}: {e.event_id} | drop={e.drop_degC:.3f} | min={e.min_time}")


if __name__ == "__main__":
    main()
