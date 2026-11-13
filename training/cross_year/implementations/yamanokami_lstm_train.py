from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import re
import pandas as pd

def is_temp_col(c: str) -> bool:
    return re.match(r"^\d+(\.\d+)?\s*m$", str(c).strip()) is not None


def parse_depth(c: str) -> float:
    return float(str(c).strip().replace(" ", "").replace("m", ""))


def build_data_only_mask(csv_file: str, drop_first_layer: bool):
    df = pd.read_csv(csv_file)

    temp_cols = [c for c in df.columns if is_temp_col(c)]
    temp_cols = sorted(temp_cols, key=parse_depth)

    if drop_first_layer:
        temp_cols = temp_cols[1:]

    mask_list = []

    for c in temp_cols:
        nan_col = f"{c}_flag_nan"
        range_col = f"{c}_flag_range"
        spike_col = f"{c}_flag_spike"
        stuck_col = f"{c}_flag_stuck"

        bad = (
            df[nan_col].fillna(False).astype(bool)
            | df[range_col].fillna(False).astype(bool)
            | df[spike_col].fillna(False).astype(bool)
            | df[stuck_col].fillna(False).astype(bool)
        )

        valid = (~bad).astype(np.float32).values
        mask_list.append(valid)

    mask = np.stack(mask_list, axis=1).astype(np.float32)
    return mask

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
# path
# =========================================================
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
# Publication packaging: filesystem routing only; numerical logic is unchanged.
DATA_ROOT = Path(os.environ.get("JOHRS_DATA_DIR", str(ROOT / "data" / "observations")))
RUN_ROOT = Path(os.environ.get("JOHRS_RUNS_DIR", str(ROOT / "outputs" / "cross_year")))
MEMBER_ROOT = Path(os.environ.get("JOHRS_MEMBER_DIR", str(RUN_ROOT)))
INFERENCE_ROOT = Path(os.environ.get("JOHRS_PREDICT_DIR", str(RUN_ROOT / "predictions")))


from src.data_loader_qc import DataConfig, load_lake_data, print_data_summary
from src.utils import set_seed, count_parameters


# =========================================================
# dataset: use observed T0
# =========================================================
class WarmupProfileDatasetQCObservedT0(Dataset):
    """
    与你当前 dataset_qc.py 不同：
    这里每个窗口都使用该窗口起点的真实温度剖面作为 T0。
    如果起点某层缺测，则用训练集平均 T0_profile 补上。
    """
    def __init__(
        self,
        meteo_norm: np.ndarray,
        meteo_raw: np.ndarray,
        temp_profile: np.ndarray,
        valid_mask: np.ndarray,
        T0_profile: np.ndarray,
        total_len: int,
        stride: int,
    ):
        super().__init__()

        assert len(meteo_norm) == len(meteo_raw) == len(temp_profile) == len(valid_mask)

        self.meteo_norm = meteo_norm.astype(np.float32)
        self.meteo_raw = meteo_raw.astype(np.float32)
        self.temp_profile = temp_profile.astype(np.float32)
        self.valid_mask = valid_mask.astype(np.float32)
        self.T0_profile = T0_profile.astype(np.float32)

        self.total_len = int(total_len)
        self.stride = int(stride)
        self.N = len(self.temp_profile)

        self.starts = np.arange(
            0,
            self.N - self.total_len + 1,
            self.stride,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        e = s + self.total_len

        x_norm = self.meteo_norm[s:e]       # (T, M)
        x_raw = self.meteo_raw[s:e]         # (T, M)
        y = self.temp_profile[s:e]          # (T, L)
        m = self.valid_mask[s:e]            # (T, L)

        # 统一 T0：所有窗口都使用同一个训练集平均温度剖面
        T0 = self.T0_profile.copy()

        return (
            torch.from_numpy(T0.astype(np.float32)),
            torch.from_numpy(x_norm),
            torch.from_numpy(x_raw),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )


# =========================================================
# pure data-driven LSTM baseline (scheme B with T0)
# =========================================================
class Seq2SeqLSTMWithT0(nn.Module):
    """
    方案 B：
      输入每个时刻: [x_t, T0]
      输出整段温度序列

    纯数据驱动，不做 autoregressive 递推。
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

        self.head = nn.Linear(hidden_dim, n_layers)

    def forward(
        self,
        T0: torch.Tensor,
        x_seq: torch.Tensor,
        x_raw_seq: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        """
        T0    : (B, L)
        x_seq : (B, T, D)
        """
        B, T, D = x_seq.shape
        assert D == self.input_dim

        T0_rep = T0.unsqueeze(1).expand(-1, T, -1)   # (B, T, L)
        x_in = torch.cat([x_seq, T0_rep], dim=-1)    # (B, T, D+L)

        h_seq, _ = self.lstm(x_in)                   # (B, T, H)
        pred = self.head(h_seq)                      # (B, T, L)

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
def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = mask.float()
    return (x * w).sum() / (w.sum() + eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    return masked_mean(err, mask)


def smoothness_loss_time(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return ((pred[:, 1:, :] - pred[:, :-1, :]) ** 2).mean()


def smoothness_loss_depth(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return ((pred[:, :, 1:] - pred[:, :, :-1]) ** 2).mean()


def range_penalty(pred: torch.Tensor, tmin: float = 0.0, tmax: float = 40.0) -> torch.Tensor:
    low = torch.relu(tmin - pred)
    high = torch.relu(pred - tmax)
    return (low ** 2).mean() + (high ** 2).mean()


# =========================================================
# train / eval
# =========================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    warmup_len: int,
    lambda_t_smooth: float,
    lambda_z_smooth: float,
    lambda_range: float,
    warmup_obs_weight: float,
    target_obs_weight: float,
    temp_min: float = 0.0,
    temp_max: float = 40.0,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_obs = 0.0
    total_warm = 0.0
    total_targ = 0.0
    total_ts = 0.0
    total_zs = 0.0
    total_rg = 0.0
    total_rmse_num = 0.0
    total_rmse_den = 0.0
    n_batch = 0

    for T0, x_norm, x_raw, y, m in dataloader:
        T0 = T0.to(device)
        x_norm = x_norm.to(device)
        y = y.to(device)
        m = m.to(device)

        pred = model(T0, x_norm)

        pred_w = pred[:, :warmup_len, :]
        y_w = y[:, :warmup_len, :]
        m_w = m[:, :warmup_len, :]

        pred_t = pred[:, warmup_len:, :]
        y_t = y[:, warmup_len:, :]
        m_t = m[:, warmup_len:, :]

        loss_w = masked_mse(pred_w, y_w, m_w)
        loss_t = masked_mse(pred_t, y_t, m_t)
        loss_obs = warmup_obs_weight * loss_w + target_obs_weight * loss_t

        loss_ts = smoothness_loss_time(pred)
        loss_zs = smoothness_loss_depth(pred)
        loss_rg = range_penalty(pred, temp_min, temp_max)

        loss = (
            loss_obs
            + lambda_t_smooth * loss_ts
            + lambda_z_smooth * loss_zs
            + lambda_range * loss_rg
        )

        se_t = (pred_t - y_t) ** 2
        rmse_num = (se_t * m_t).sum().item()
        rmse_den = m_t.sum().item()

        total_loss += loss.item()
        total_obs += loss_obs.item()
        total_warm += loss_w.item()
        total_targ += loss_t.item()
        total_ts += loss_ts.item()
        total_zs += loss_zs.item()
        total_rg += loss_rg.item()
        total_rmse_num += rmse_num
        total_rmse_den += rmse_den
        n_batch += 1

    out = {
        "loss": total_loss / max(n_batch, 1),
        "obs": total_obs / max(n_batch, 1),
        "warmup_obs": total_warm / max(n_batch, 1),
        "target_obs": total_targ / max(n_batch, 1),
        "t_smooth": total_ts / max(n_batch, 1),
        "z_smooth": total_zs / max(n_batch, 1),
        "range": total_rg / max(n_batch, 1),
        "rmse": math.sqrt(total_rmse_num / max(total_rmse_den, 1.0)),
        "n_valid_target_points": float(total_rmse_den),
    }
    return out


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_len: int,
    lambda_t_smooth: float,
    lambda_z_smooth: float,
    lambda_range: float,
    warmup_obs_weight: float,
    target_obs_weight: float,
    grad_clip: float | None = None,
    temp_min: float = 0.0,
    temp_max: float = 40.0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_obs = 0.0
    total_warm = 0.0
    total_targ = 0.0
    total_ts = 0.0
    total_zs = 0.0
    total_rg = 0.0
    n_batch = 0

    for T0, x_norm, x_raw, y, m in dataloader:
        T0 = T0.to(device)
        x_norm = x_norm.to(device)
        y = y.to(device)
        m = m.to(device)

        optimizer.zero_grad()

        pred = model(T0, x_norm)

        pred_w = pred[:, :warmup_len, :]
        y_w = y[:, :warmup_len, :]
        m_w = m[:, :warmup_len, :]

        pred_t = pred[:, warmup_len:, :]
        y_t = y[:, warmup_len:, :]
        m_t = m[:, warmup_len:, :]

        loss_w = masked_mse(pred_w, y_w, m_w)
        loss_t = masked_mse(pred_t, y_t, m_t)
        loss_obs = warmup_obs_weight * loss_w + target_obs_weight * loss_t

        loss_ts = smoothness_loss_time(pred)
        loss_zs = smoothness_loss_depth(pred)
        loss_rg = range_penalty(pred, temp_min, temp_max)

        loss = (
            loss_obs
            + lambda_t_smooth * loss_ts
            + lambda_z_smooth * loss_zs
            + lambda_range * loss_rg
        )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_obs += loss_obs.item()
        total_warm += loss_w.item()
        total_targ += loss_t.item()
        total_ts += loss_ts.item()
        total_zs += loss_zs.item()
        total_rg += loss_rg.item()
        n_batch += 1

    out = {
        "loss": total_loss / max(n_batch, 1),
        "obs": total_obs / max(n_batch, 1),
        "warmup_obs": total_warm / max(n_batch, 1),
        "target_obs": total_targ / max(n_batch, 1),
        "t_smooth": total_ts / max(n_batch, 1),
        "z_smooth": total_zs / max(n_batch, 1),
        "range": total_rg / max(n_batch, 1),
    }
    return out


@torch.no_grad()
def predict_loader(model: nn.Module, dataloader, device: torch.device):
    model.eval()

    pred_list: List[np.ndarray] = []
    targ_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    for T0, x_norm, x_raw, y, m in dataloader:
        T0 = T0.to(device)
        x_norm = x_norm.to(device)

        pred = model(T0, x_norm)

        pred_list.append(pred.cpu().numpy())
        targ_list.append(y.numpy())
        mask_list.append(m.numpy())

    return (
        np.concatenate(pred_list, axis=0),
        np.concatenate(targ_list, axis=0),
        np.concatenate(mask_list, axis=0),
    )


def train_model(
    model: nn.Module,
    dl_tr,
    dl_va,
    device: torch.device,
    save_dir: str,
    warmup_len: int,
    target_len: int,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_t_smooth: float = 0.0,
    lambda_z_smooth: float = 0.0,
    lambda_range: float = 0.0,
    warmup_obs_weight: float = 0.3,
    target_obs_weight: float = 1.0,
    grad_clip: float | None = 5.0,
    temp_min: float = 0.0,
    temp_max: float = 40.0,
):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history: Dict[str, List[dict]] = {
        "train": [],
        "val": [],
    }

    best_val = float("inf")
    best_ckpt = os.path.join(save_dir, "best_model.pt")

    print("\n===== training LSTM =====")
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            dataloader=dl_tr,
            optimizer=optimizer,
            device=device,
            warmup_len=warmup_len,
            lambda_t_smooth=lambda_t_smooth,
            lambda_z_smooth=lambda_z_smooth,
            lambda_range=lambda_range,
            warmup_obs_weight=warmup_obs_weight,
            target_obs_weight=target_obs_weight,
            grad_clip=grad_clip,
            temp_min=temp_min,
            temp_max=temp_max,
        )

        va = evaluate_model(
            model=model,
            dataloader=dl_va,
            device=device,
            warmup_len=warmup_len,
            lambda_t_smooth=lambda_t_smooth,
            lambda_z_smooth=lambda_z_smooth,
            lambda_range=lambda_range,
            warmup_obs_weight=warmup_obs_weight,
            target_obs_weight=target_obs_weight,
            temp_min=temp_min,
            temp_max=temp_max,
        )

        history["train"].append({"epoch": epoch, **tr})
        history["val"].append({"epoch": epoch, **va})

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_ckpt)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{epoch:03d}] "
                f"train={tr['loss']:.6f} "
                f"val={va['loss']:.6f} "
                f"obs={va['obs']:.6f} "
                f"rmse={va['rmse']:.4f}"
            )

    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    pred_val, targ_val, mask_val = predict_loader(model, dl_va, device)

    np.savez_compressed(
        os.path.join(save_dir, "val_predictions_best.npz"),
        pred=pred_val,
        target=targ_val,
        mask=mask_val,
        warmup_len=warmup_len,
        target_len=target_len,
    )

    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n===== done =====")
    print("saved:", best_ckpt)
    print("saved:", os.path.join(save_dir, "val_predictions_best.npz"))
    print("saved:", os.path.join(save_dir, "history.json"))


# =========================================================
# main
# =========================================================
def main(seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # =========================================================
    # config
    # =========================================================
    CSV_FILE = str(DATA_ROOT / "yamanokami_2025_aligned_qc_wide.csv")
    SAVE_DIR = str(RUN_ROOT / f"seed{seed}")

    WARMUP_LEN = 1008
    TARGET_LEN = 288
    TOTAL_LEN = WARMUP_LEN + TARGET_LEN
    STRIDE = 24

    BATCH_SIZE = 64
    NUM_WORKERS = 0

    HIDDEN_DIM = 64
    NUM_LSTM_LAYERS = 1
    DROPOUT = 0.0

    EPOCHS = 200
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    # pure data-driven baseline，建议先设为 0
    LAMBDA_T_SMOOTH = 0.0
    LAMBDA_Z_SMOOTH = 0.0
    LAMBDA_RANGE = 0.0

    WARMUP_OBS_WEIGHT = 0.3
    TARGET_OBS_WEIGHT = 1.0

    TEMP_MIN = 0.0
    TEMP_MAX = 40.0

    # =========================================================
    # data
    # =========================================================
    cfg = DataConfig(
        csv_file=CSV_FILE,
        norm_end_idx=30000,
        add_doy=True,
        drop_first_temp_layer=True,
    )

    data = load_lake_data(cfg)
    print("\n===== override mask: data-only (no physics filtering) =====")

    new_mask = build_data_only_mask(
        csv_file=CSV_FILE,
        drop_first_layer=True  
    )
    
    if new_mask.shape != data.valid_mask.shape:
        raise ValueError(f"mask shape mismatch: {new_mask.shape} vs {data.valid_mask.shape}")
    
    data.valid_mask = new_mask
    
    print("new valid ratio =", float(data.valid_mask.mean()))

    data.meteo_raw[:, :6] = fill_nan_1d_linear(data.meteo_raw[:, :6])
    meteo_mean = data.meteo_raw[:, :6].mean(axis=0, keepdims=True).astype(np.float32)
    meteo_std = data.meteo_raw[:, :6].std(axis=0, keepdims=True).astype(np.float32)
    data.meteo_norm[:, :6] = (
        (data.meteo_raw[:, :6] - meteo_mean) / meteo_std
    ).astype(np.float32)

    data.meteo_raw[:, :6] = fill_nan_1d_linear(data.meteo_raw[:, :6])
    meteo_mean = data.meteo_raw[:, :6].mean(axis=0, keepdims=True).astype(np.float32)
    meteo_std = data.meteo_raw[:, :6].std(axis=0, keepdims=True).astype(np.float32)
    data.meteo_norm[:, :6] = (
        (data.meteo_raw[:, :6] - meteo_mean) / meteo_std
    ).astype(np.float32)
    
    print_data_summary(data)

    n_total = len(data.time)
    split_idx = int(n_total * 0.9)

    print("total rows =", n_total)
    print("split idx  =", split_idx)

    if split_idx < TOTAL_LEN:
        raise ValueError(f"训练集长度不足。split_idx={split_idx}, total_len={TOTAL_LEN}")
    if (n_total - split_idx) < TOTAL_LEN:
        raise ValueError(f"验证集长度不足。val_len={n_total - split_idx}, total_len={TOTAL_LEN}")

    # 90% train / 10% val
    meteo_norm_train = data.meteo_norm[:split_idx]
    meteo_raw_train = data.meteo_raw[:split_idx]
    temp_train = data.temp_profile[:split_idx]
    mask_train = data.valid_mask[:split_idx]

    val_start = split_idx - (TOTAL_LEN - 1)
    meteo_norm_val = data.meteo_norm[val_start:]
    meteo_raw_val = data.meteo_raw[val_start:]
    temp_val = data.temp_profile[val_start:]
    mask_val = data.valid_mask[val_start:]

    # 训练集平均剖面仅用于起点缺测补值
    valid_sum = mask_train.sum(axis=0)
    temp_sum = (temp_train * mask_train).sum(axis=0)
    T0_profile_np = temp_sum / np.clip(valid_sum, 1e-6, None)
    T0_profile_np = T0_profile_np.astype(np.float32)

    print("fallback T0_profile shape =", T0_profile_np.shape)
    print("fallback T0_profile =", T0_profile_np)

    ds_tr = WarmupProfileDatasetQCObservedT0(
        meteo_norm=meteo_norm_train,
        meteo_raw=meteo_raw_train,
        temp_profile=temp_train,
        valid_mask=mask_train,
        T0_profile=T0_profile_np,
        total_len=TOTAL_LEN,
        stride=STRIDE,
    )

    ds_va = WarmupProfileDatasetQCObservedT0(
        meteo_norm=meteo_norm_val,
        meteo_raw=meteo_raw_val,
        temp_profile=temp_val,
        valid_mask=mask_val,
        T0_profile=T0_profile_np,
        total_len=TOTAL_LEN,
        stride=STRIDE,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    dl_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    print("train windows =", len(ds_tr))
    print("val windows   =", len(ds_va))

    # =========================================================
    # model
    # =========================================================
    model = Seq2SeqLSTMWithT0(
        input_dim=data.meteo_norm.shape[1],
        n_layers=temp_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print("n_layers  =", temp_train.shape[1])
    print("input_dim =", data.meteo_norm.shape[1])
    print("params    =", count_parameters(model))

    # =========================================================
    # train
    # =========================================================
    train_model(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        device=device,
        save_dir=SAVE_DIR,
        warmup_len=WARMUP_LEN,
        target_len=TARGET_LEN,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        lambda_t_smooth=LAMBDA_T_SMOOTH,
        lambda_z_smooth=LAMBDA_Z_SMOOTH,
        lambda_range=LAMBDA_RANGE,
        warmup_obs_weight=WARMUP_OBS_WEIGHT,
        target_obs_weight=TARGET_OBS_WEIGHT,
        grad_clip=GRAD_CLIP,
        temp_min=TEMP_MIN,
        temp_max=TEMP_MAX,
    )



if __name__ == "__main__":
    raise SystemExit("Use: python training/cross_year/driver.py --help")
