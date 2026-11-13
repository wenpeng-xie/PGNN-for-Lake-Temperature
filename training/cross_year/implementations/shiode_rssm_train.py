from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
# Publication packaging: filesystem routing only; numerical logic is unchanged.
DATA_ROOT = Path(os.environ.get("JOHRS_DATA_DIR", str(ROOT / "data" / "observations")))
RUN_ROOT = Path(os.environ.get("JOHRS_RUNS_DIR", str(ROOT / "outputs" / "cross_year")))
MEMBER_ROOT = Path(os.environ.get("JOHRS_MEMBER_DIR", str(RUN_ROOT)))
INFERENCE_ROOT = Path(os.environ.get("JOHRS_PREDICT_DIR", str(RUN_ROOT / "predictions")))


import numpy as np
import torch
from torch.utils.data import DataLoader

# =========================================================
# path
# =========================================================

from src.data_loader_qc import DataConfig, load_lake_data, print_data_summary
from src.dataset_qc import WarmupProfileDatasetQC
from src.trainer_qc import train_model
from src.utils import count_parameters, set_seed
from src.models import ResidualSSM

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

def main(seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # =========================================================
    # config
    # =========================================================
    CSV_FILE = str(DATA_ROOT / "shiode_2024_aligned_qc_wide.csv")
    SAVE_DIR = str(RUN_ROOT / f"seed{seed}")

    # 序列长度设置
    WARMUP_LEN = 1008
    TARGET_LEN = 288
    TOTAL_LEN = WARMUP_LEN + TARGET_LEN
    STRIDE = 24

    # dataloader
    BATCH_SIZE = 256
    NUM_WORKERS = 0

    # model
    HIDDEN_DIM = 64
    UPDATE_SCALE = 0.05

    # training
    EPOCHS = 200
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    LAMBDA_T_SMOOTH = 1e-4
    LAMBDA_Z_SMOOTH = 1e-4
    LAMBDA_RANGE = 1e-3

    # 观测损失权重
    WARMUP_OBS_WEIGHT = 0.2
    TARGET_OBS_WEIGHT = 1.0

    TEMP_MIN = 0.0
    TEMP_MAX = 40.0

    # =========================================================
    # data
    # =========================================================
    cfg = DataConfig(
        csv_file=CSV_FILE,
        norm_end_idx=30000,          # 如果总长度不足 30000，loader 内部会自动截断
        add_doy=True,
        drop_first_temp_layer=False,  # 你现在这套 loader 默认可选去掉第一层
    )

    data = load_lake_data(cfg)

    print("\n===== override mask: data-only (no physics filtering) =====")

    new_mask = build_data_only_mask(
        csv_file=CSV_FILE,
        drop_first_layer=False  # ⚠️ 你这里是 False，要一致
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

    print_data_summary(data)

    n_total = len(data.time)
    split_idx = int(n_total * 0.9)

    print("total rows =", n_total)
    print("split idx  =", split_idx)

    if split_idx < TOTAL_LEN:
        raise ValueError(
            f"训练集长度不足。split_idx={split_idx}, 但 total_len={TOTAL_LEN}"
        )
    if (n_total - split_idx) < TOTAL_LEN:
        raise ValueError(
            f"验证集长度不足。val_len={n_total - split_idx}, 但 total_len={TOTAL_LEN}"
        )

    # -----------------------------
    # split: 90% train / 10% val
    # -----------------------------
    meteo_norm_train = data.meteo_norm[:split_idx]
    meteo_raw_train = data.meteo_raw[:split_idx]
    temp_train = data.temp_profile[:split_idx]
    mask_train = data.valid_mask[:split_idx]


    # 为了保证 val 窗口能完整构造，这里从 split_idx - (TOTAL_LEN - 1) 开始切
    val_start = split_idx - (TOTAL_LEN - 1)

    meteo_norm_val = data.meteo_norm[val_start:]
    meteo_raw_val = data.meteo_raw[val_start:]
    temp_val = data.temp_profile[val_start:]
    mask_val = data.valid_mask[val_start:]

    # =========================================================
    # T0: 训练集平均温度剖面
    # 只用训练集有效值计算
    # =========================================================
    valid_sum = mask_train.sum(axis=0)
    temp_sum = (temp_train * mask_train).sum(axis=0)
    T0_profile_np = temp_sum / np.clip(valid_sum, 1e-6, None)
    T0_profile_np = T0_profile_np.astype(np.float32)

    print("T0_profile shape =", T0_profile_np.shape)
    print("T0_profile =", T0_profile_np)

    # =========================================================
    # dataset / dataloader
    # =========================================================
    ds_tr = WarmupProfileDatasetQC(
        meteo_norm=meteo_norm_train,
        meteo_raw=meteo_raw_train,
        temp_profile=temp_train,
        valid_mask=mask_train,
        T0_profile=T0_profile_np,
        total_len=TOTAL_LEN,
        stride=STRIDE,
    )

    ds_va = WarmupProfileDatasetQC(
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
    model = ResidualSSM(
        input_dim=data.meteo_norm.shape[1],
        n_layers=temp_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        update_scale=UPDATE_SCALE,
    ).to(device)

    print("n_layers  =", temp_train.shape[1])
    print("input_dim =", data.meteo_norm.shape[1])
    print("params    =", count_parameters(model))

    # =========================================================
    # train
    # =========================================================
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_model(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        device=device,
        save_dir=SAVE_DIR,
        warmup_len=WARMUP_LEN,
        target_len=TARGET_LEN,
        T0_profile_np=T0_profile_np,
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
