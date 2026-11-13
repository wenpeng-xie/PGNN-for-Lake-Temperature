from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
# Publication packaging: filesystem routing only; numerical logic is unchanged.
DATA_ROOT = Path(os.environ.get("JOHRS_DATA_DIR", str(ROOT / "data" / "observations")))
RUN_ROOT = Path(os.environ.get("JOHRS_RUNS_DIR", str(ROOT / "outputs" / "cross_year")))
MEMBER_ROOT = Path(os.environ.get("JOHRS_MEMBER_DIR", str(RUN_ROOT)))
INFERENCE_ROOT = Path(os.environ.get("JOHRS_PREDICT_DIR", str(RUN_ROOT / "predictions")))


from src.data_loader_qc import (
    DataConfig,
    load_lake_data,
    load_lake_data_with_given_norm,
    print_data_summary,
)

# =========================================================
# helpers
# =========================================================
def fill_nan_1d_linear(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    n, d = arr.shape
    idx = np.arange(n)

    for j in range(d):
        col = arr[:, j]
        valid = np.isfinite(col)
        if valid.all():
            continue
        if valid.sum() == 0:
            raise ValueError(f"column {j} is all NaN")
        arr[:, j] = np.interp(idx, idx[valid], col[valid]).astype(np.float32)
    return arr
# =========================================================
# config
# =========================================================
TRAIN_CSV_2025 = str(DATA_ROOT / "shiode_2024_aligned_qc_wide.csv")
TEST_CSV = str(DATA_ROOT / "shiode_2025_aligned_qc_wide.csv")

WARMUP_LEN = 1008
TARGET_LEN = 288
TOTAL_LEN = WARMUP_LEN + TARGET_LEN
STRIDE_EVAL = 24

MAX_DEPTH_DIFF = 0.10

HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 1
DROPOUT = 0.0


# =========================================================
# model
# =========================================================
class StructureAwareLSTMWithT0(nn.Module):
    """
    Input:
        T0    : (B, L)
        x_seq : (B, T, D)

    Output:
        pred  : (B, T, L)

    Same interface as your original Seq2SeqLSTMWithT0.
    """
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim: int = 64,
        num_lstm_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        self.total_input_dim = input_dim + n_layers

        self.lstm = nn.LSTM(
            input_size=self.total_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.temp_head = nn.Linear(hidden_dim, n_layers)

    def forward(
        self,
        T0: torch.Tensor,
        x_seq: torch.Tensor,
        x_raw_seq: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        B, T, D = x_seq.shape
        assert D == self.input_dim

        T0_rep = T0.unsqueeze(1).expand(-1, T, -1)
        x_in = torch.cat([x_seq, T0_rep], dim=-1)

        h_seq, _ = self.lstm(x_in)
        pred = self.temp_head(h_seq)

        if not return_diagnostics:
            return pred

        diagnostics = {
            "pred_mean": pred.mean().detach().item(),
            "pred_std": pred.std().detach().item(),
        }
        return pred, diagnostics


# =========================================================
# helpers
# =========================================================
def build_nearest_depth_mapping(model_depths, obs_depths, max_diff=0.10):
    model_depths = np.asarray(model_depths, dtype=float)
    obs_depths = np.asarray(obs_depths, dtype=float)

    pairs = []
    for obs_idx, d_obs in enumerate(obs_depths):
        diffs = np.abs(model_depths - d_obs)
        model_idx = int(np.argmin(diffs))
        diff = float(diffs[model_idx])
        if diff <= max_diff:
            pairs.append((obs_idx, model_idx, float(d_obs), float(model_depths[model_idx]), diff))
        else:
            print(f"[skip] obs depth {d_obs:.3f} m -> no model depth within {max_diff:.3f} m")
    return pairs


def print_depth_mapping(pairs):
    print("\n===== nearest depth mapping =====")
    for obs_idx, model_idx, d_obs, d_model, diff in pairs:
        print(
            f"obs_idx={obs_idx:02d} ({d_obs:.3f} m) "
            f"-> model_idx={model_idx:02d} ({d_model:.3f} m) "
            f"|diff|={diff:.3f} m"
        )


def build_model_grid_T0_from_obs(
    model_depths,
    obs_depths,
    obs_T0,
    obs_mask0,
    fallback_profile,
    max_diff=0.10,
):
    model_depths = np.asarray(model_depths, dtype=float)
    obs_depths = np.asarray(obs_depths, dtype=float)
    obs_T0 = np.asarray(obs_T0, dtype=np.float32)
    obs_mask0 = np.asarray(obs_mask0, dtype=np.float32)
    fallback_profile = np.asarray(fallback_profile, dtype=np.float32)

    T0_model = fallback_profile.copy()

    for model_idx, d_model in enumerate(model_depths):
        diffs = np.abs(obs_depths - d_model)
        obs_idx = int(np.argmin(diffs))
        diff = float(diffs[obs_idx])

        if diff <= max_diff and obs_mask0[obs_idx] > 0.5 and np.isfinite(obs_T0[obs_idx]):
            T0_model[model_idx] = obs_T0[obs_idx]

    return T0_model.astype(np.float32)


def calc_metrics(pred, obs, mask):
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(obs)

    p = pred[valid]
    o = obs[valid]

    if len(p) == 0:
        return {"RMSE": np.nan, "R": np.nan, "Bias": np.nan, "NSE": np.nan, "N": 0}

    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    bias = float(np.mean(p - o))

    if len(p) >= 2 and np.std(p) > 0 and np.std(o) > 0:
        r = float(np.corrcoef(p, o)[0, 1])
    else:
        r = np.nan

    denom = np.sum((o - np.mean(o)) ** 2)
    nse = float(1.0 - np.sum((p - o) ** 2) / denom) if denom > 0 else np.nan

    return {"RMSE": rmse, "R": r, "Bias": bias, "NSE": nse, "N": int(len(p))}


def freshwater_density_np(T):
    T = np.asarray(T, dtype=float)
    return 1000.0 * (
        1.0
        - ((T + 288.9414) / (508929.2 * (T + 68.12963)))
        * (T - 3.9863) ** 2
    )


def calc_structure_series_np(T, mask, depths, min_valid_fraction=0.7):
    """
    Calculate STD, Gmean, and H from T[time, depth].

    STD: vertical temperature standard deviation, degC
    Gmean: mean absolute vertical thermal gradient, degC/m
    H: water-column heat content per unit area, MJ/m2
    """
    T = np.asarray(T, dtype=float)
    mask = (np.asarray(mask) > 0.5) & np.isfinite(T)
    z = np.asarray(depths, dtype=float)

    n_time, n_depth = T.shape
    STD = np.full(n_time, np.nan, dtype=float)
    Gmean = np.full(n_time, np.nan, dtype=float)
    H = np.full(n_time, np.nan, dtype=float)

    dz_layer = np.gradient(z)

    for t in range(n_time):
        valid = mask[t]
        if valid.mean() < min_valid_fraction or valid.sum() < 2:
            continue

        row = T[t].copy()
        row_valid = row[valid]
        z_valid = z[valid]

        STD[t] = np.nanstd(row_valid)

        # interpolate to full model/eval grid for comparable G/H
        filled = np.interp(z, z_valid, row_valid)

        grad = np.diff(filled) / np.diff(z)
        Gmean[t] = np.nanmean(np.abs(grad))

        rho = freshwater_density_np(filled)
        H[t] = np.sum(rho * 4186.0 * filled * dz_layer) / 1e6

    return {"STD": STD, "Gmean": Gmean, "H": H}


def eval_structure_metrics(pred, obs, mask, depths):
    pred_struct = calc_structure_series_np(pred, mask, depths)
    obs_struct = calc_structure_series_np(obs, mask, depths)

    out = {}
    for key in ["STD", "Gmean", "H"]:
        m = np.isfinite(pred_struct[key]) & np.isfinite(obs_struct[key])
        out[key] = calc_metrics(
            pred=pred_struct[key][m],
            obs=obs_struct[key][m],
            mask=np.ones(np.sum(m), dtype=np.float32),
        )
    return out


def overlap_average_target_only(pred_windows, starts, warmup_len, total_len, n_total):
    """
    pred_windows: (Nw, total_len, L)
    starts: (Nw,)
    只把每个窗口 warmup 之后的 target 段累计到最终输出中
    """
    n_layers = pred_windows.shape[-1]

    sum_arr = np.zeros((n_total, n_layers), dtype=np.float64)
    cnt_arr = np.zeros((n_total, n_layers), dtype=np.float64)

    for w, s in enumerate(starts):
        s_use = s + warmup_len
        e_use = s + total_len

        # 只取当前窗口的 target 段
        pred_use = pred_windows[w, warmup_len:, :]   # (target_len, L)

        sum_arr[s_use:e_use] += pred_use
        cnt_arr[s_use:e_use] += 1.0

    pred_mean = sum_arr / np.clip(cnt_arr, 1e-12, None)
    return pred_mean.astype(np.float32), cnt_arr.astype(np.float32)


def select_matched_layers_for_eval(pred_full, obs_full, mask_full, pairs):
    obs_indices = [p[0] for p in pairs]
    model_indices = [p[1] for p in pairs]
    obs_depths_sel = np.array([p[2] for p in pairs], dtype=np.float32)
    model_depths_sel = np.array([p[3] for p in pairs], dtype=np.float32)

    pred_sel = pred_full[:, model_indices]
    obs_sel = obs_full[:, obs_indices]
    mask_sel = mask_full[:, obs_indices]

    return pred_sel, obs_sel, mask_sel, obs_depths_sel, model_depths_sel


def plot_timeseries_by_layer(time, obs, pred, mask, obs_depths, model_depths, warmup_len, out_png, max_layers_per_fig=4):
    n_layers = len(obs_depths)
    start = 0

    while start < n_layers:
        end = min(start + max_layers_per_fig, n_layers)
        n_sub = end - start

        fig, axes = plt.subplots(n_sub, 1, figsize=(16, 3.2 * n_sub), sharex=True)
        if n_sub == 1:
            axes = [axes]

        for ax, i in zip(axes, range(start, end)):
            obs_i = obs[:, i].copy()
            obs_i[mask[:, i] < 0.5] = np.nan

            ax.plot(time, obs_i, label=f"Obs ({obs_depths[i]:.2f}m)")
            ax.plot(time, pred[:, i], label=f"Pred ({model_depths[i]:.2f}m)")
            ax.axvline(time[warmup_len], linestyle="--")
            ax.set_ylabel(f"L{i+1}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        save_path = out_png.replace(".png", f"_layers_{start+1}_{end}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        start = end


def plot_scatter(pred, obs, mask, out_png):
    valid = mask > 0.5
    p = pred[valid]
    o = obs[valid]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(o, p, s=6, alpha=0.5)

    if len(o) > 0:
        vmin = float(np.nanmin([o.min(), p.min()]))
        vmax = float(np.nanmax([o.max(), p.max()]))
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("LSTM: Pred vs Obs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(obs, pred, mask, warmup_len, out_png):
    obs_plot = obs.copy()
    obs_plot[mask < 0.5] = np.nan
    err = pred - obs
    err[mask < 0.5] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    im0 = axes[0].imshow(obs_plot.T, aspect="auto", origin="upper", interpolation="nearest")
    axes[0].axvline(warmup_len, linestyle="--")
    axes[0].set_title("Observed")
    fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.02)

    im1 = axes[1].imshow(pred.T, aspect="auto", origin="upper", interpolation="nearest")
    axes[1].axvline(warmup_len, linestyle="--")
    axes[1].set_title("Predicted")
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.02)

    vmax = np.nanmax(np.abs(err)) if np.isfinite(err).any() else 1.0
    im2 = axes[2].imshow(err.T, aspect="auto", origin="upper", interpolation="nearest", vmin=-vmax, vmax=vmax)
    axes[2].axvline(warmup_len, linestyle="--")
    axes[2].set_title("Error (Pred - Obs)")
    fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# main
# =========================================================
def main(seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    CKPT_PATH = str(MEMBER_ROOT / "best_model.pt")
    OUT_DIR = str(INFERENCE_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- load 2025 train data --------
    cfg_2025 = DataConfig(
        csv_file=TRAIN_CSV_2025,
        norm_end_idx=30000,
        add_doy=True,
        drop_first_temp_layer=False,
    )
    data_2025 = load_lake_data(cfg_2025)
    
    print("\n===== 2025 data summary (model grid source) =====")
    print_data_summary(data_2025)
    
    model_depths = data_2025.depths.astype(np.float32)
    
    # 2025 fallback T0
    valid_sum_2025 = data_2025.valid_mask.sum(axis=0)
    temp_sum_2025 = (data_2025.temp_profile * data_2025.valid_mask).sum(axis=0)
    fallback_profile_2025 = temp_sum_2025 / np.clip(valid_sum_2025, 1e-6, None)
    fallback_profile_2025 = fallback_profile_2025.astype(np.float32)
    
    # 2025 mean/std for normalization
    # 注意：这里要按原始 meteo 列数，只取前 6 列，不包含 doy
    meteo_raw_2025_base = data_2025.meteo_raw[:, :6]
    meteo_raw_2025_base = fill_nan_1d_linear(meteo_raw_2025_base)
    
    meteo_mean_2025 = meteo_raw_2025_base.mean(axis=0, keepdims=True).astype(np.float32)
    meteo_std_2025 = meteo_raw_2025_base.std(axis=0, keepdims=True).astype(np.float32)
    meteo_std_2025 = np.where(meteo_std_2025 < 1e-6, 1.0, meteo_std_2025)
    
    # -------- load 2024 test data using 2025 normalization --------
    cfg_test = DataConfig(
        csv_file=TEST_CSV,
        norm_end_idx=30000,
        add_doy=True,
        drop_first_temp_layer=False,
    )
    data_test = load_lake_data_with_given_norm(
        cfg=cfg_test,
        meteo_mean=meteo_mean_2025,
        meteo_std=meteo_std_2025,
    )
    
    # 只修前6列原始气象变量
    data_test.meteo_raw[:, :6] = fill_nan_1d_linear(data_test.meteo_raw[:, :6])
    
    # 再重新算 meteo_norm 的前6列
    data_test.meteo_norm[:, :6] = (
        (data_test.meteo_raw[:, :6] - meteo_mean_2025) / meteo_std_2025
    ).astype(np.float32)
    
    print("\n===== test data summary (using 2025 normalization stats) =====")
    print_data_summary(data_test)
    print("\n===== input NaN check =====")
    print("NaN in meteo_norm:", np.isnan(data_test.meteo_norm).sum())
    print("NaN in meteo_raw :", np.isnan(data_test.meteo_raw).sum())
    print("NaN in temp_profile:", np.isnan(data_test.temp_profile).sum())
    print("NaN in valid_mask:", np.isnan(data_test.valid_mask).sum())

    for j in range(data_test.meteo_norm.shape[1]):
        n_nan = np.isnan(data_test.meteo_norm[:, j]).sum()
        print(f"meteo_norm col {j} NaN count = {n_nan}")

    n_total = len(data_test.time)
    if n_total < TOTAL_LEN:
        raise ValueError(f"test length {n_total} < TOTAL_LEN {TOTAL_LEN}")

    # -------- nearest depth matching --------
    pairs = build_nearest_depth_mapping(
        model_depths=model_depths,
        obs_depths=data_test.depths,
        max_diff=MAX_DEPTH_DIFF,
    )
    print_depth_mapping(pairs)

    if len(pairs) == 0:
        raise RuntimeError("No matched depths.")

    # -------- build true T0 on model grid --------
    obs_T0 = data_test.temp_profile[0]
    obs_mask0 = data_test.valid_mask[0]

    T0_model_np = build_model_grid_T0_from_obs(
        model_depths=model_depths,
        obs_depths=data_test.depths,
        obs_T0=obs_T0,
        obs_mask0=obs_mask0,
        fallback_profile=fallback_profile_2025,
        max_diff=MAX_DEPTH_DIFF,
    )

    print("\n===== evaluation T0 on model grid =====")
    print("model_depths =", model_depths.tolist())
    print("T0_model_np  =", T0_model_np.tolist())

    # -------- build model --------
    model = StructureAwareLSTMWithT0(
        input_dim=data_2025.meteo_norm.shape[1],
        n_layers=len(model_depths),
        hidden_dim=HIDDEN_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print("\nLoaded checkpoint:", CKPT_PATH)

    # -------- sliding windows with stride=1 --------
    starts = np.arange(0, n_total - TOTAL_LEN + 1, STRIDE_EVAL, dtype=np.int64)

    pred_windows = []

    with torch.no_grad():
        for s in tqdm(starts, desc="LSTM evaluating"):
            e = s + TOTAL_LEN
    
            x_norm_win = data_test.meteo_norm[s:e].astype(np.float32)
            x_norm_win = torch.from_numpy(x_norm_win).unsqueeze(0).to(device)
    
            T0 = torch.from_numpy(T0_model_np).unsqueeze(0).to(device)
    
            pred_win = model(T0, x_norm_win).cpu().numpy()[0]
            pred_windows.append(pred_win)

    pred_windows = np.stack(pred_windows, axis=0)

    # -------- overlap average --------
    pred_full_model_grid, count_full = overlap_average_target_only(
        pred_windows=pred_windows,
        starts=starts,
        warmup_len=WARMUP_LEN,
        total_len=TOTAL_LEN,
        n_total=n_total,
    )

    # -------- select matched layers only --------
    pred_sel, obs_sel, mask_sel, obs_depths_sel, model_depths_sel = select_matched_layers_for_eval(
        pred_full=pred_full_model_grid,
        obs_full=data_test.temp_profile.astype(np.float32),
        mask_full=data_test.valid_mask.astype(np.float32),
        pairs=pairs,
    )

    model_indices = [p[1] for p in pairs]
    pred_valid_mask = (count_full > 0).astype(np.float32)
    
    mask_sel = mask_sel * pred_valid_mask[:, model_indices]

    # -------- metrics --------
    metrics_all = calc_metrics(
        pred=pred_sel[WARMUP_LEN:],
        obs=obs_sel[WARMUP_LEN:],
        mask=mask_sel[WARMUP_LEN:],
    )

    layer_metrics = {}
    for i in range(len(obs_depths_sel)):
        key = f"obs_{obs_depths_sel[i]:.2f}m__pred_{model_depths_sel[i]:.2f}m"
        layer_metrics[key] = calc_metrics(
            pred=pred_sel[WARMUP_LEN:, i:i+1],
            obs=obs_sel[WARMUP_LEN:, i:i+1],
            mask=mask_sel[WARMUP_LEN:, i:i+1],
        )

    print("\n===== overall metrics after warmup =====")
    for k, v in metrics_all.items():
        print(k, "=", v)

    structure_metrics = eval_structure_metrics(
        pred=pred_sel[WARMUP_LEN:],
        obs=obs_sel[WARMUP_LEN:],
        mask=mask_sel[WARMUP_LEN:],
        depths=obs_depths_sel,
    )

    print("\n===== structure metrics after warmup =====")
    for target, mm in structure_metrics.items():
        print(target, mm)

    # -------- save --------
    np.savez_compressed(
        os.path.join(OUT_DIR, "lstm_overlap_avg_predictions.npz"),
        time=data_test.time,
        pred_full_model_grid=pred_full_model_grid,
        pred_sel=pred_sel,
        obs_sel=obs_sel,
        mask_sel=mask_sel,
        counts=count_full,
        model_depths=model_depths,
        obs_depths=data_test.depths,
        matched_obs_depths=obs_depths_sel,
        matched_model_depths=model_depths_sel,
        warmup_len=WARMUP_LEN,
        starts=starts,
        T0_model=T0_model_np,
    )

    with open(os.path.join(OUT_DIR, "metrics_overall.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    with open(os.path.join(OUT_DIR, "metrics_by_layer.json"), "w", encoding="utf-8") as f:
        json.dump(layer_metrics, f, indent=2)

    with open(os.path.join(OUT_DIR, "metrics_structure.json"), "w", encoding="utf-8") as f:
        json.dump(structure_metrics, f, indent=2)

    # -------- plots --------
    plot_timeseries_by_layer(
        time=data_test.time,
        obs=obs_sel,
        pred=pred_sel,
        mask=mask_sel,
        obs_depths=obs_depths_sel,
        model_depths=model_depths_sel,
        warmup_len=WARMUP_LEN,
        out_png=os.path.join(OUT_DIR, "timeseries_by_layer.png"),
        max_layers_per_fig=4,
    )

    plot_scatter(
        pred=pred_sel[WARMUP_LEN:],
        obs=obs_sel[WARMUP_LEN:],
        mask=mask_sel[WARMUP_LEN:],
        out_png=os.path.join(OUT_DIR, "scatter_after_warmup.png"),
    )

    plot_heatmaps(
        obs=obs_sel,
        pred=pred_sel,
        mask=mask_sel,
        warmup_len=WARMUP_LEN,
        out_png=os.path.join(OUT_DIR, "heatmaps.png"),
    )

    print("\nSaved results to:", OUT_DIR)



if __name__ == "__main__":
    raise SystemExit("Use: python training/cross_year/driver.py --help")
