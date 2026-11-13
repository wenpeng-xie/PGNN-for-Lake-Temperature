from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    csv_file: str

    meteo_cols: tuple[str, ...] = (
        "solar", "precip", "wind", "air_temp", "rh", "pressure"
    )

    norm_end_idx: int = 30000
    add_doy: bool = True
    doy_period: float = 366.0

    drop_first_temp_layer: bool = True


@dataclass
class LakeData:
    time: np.ndarray
    meteo_raw: np.ndarray
    meteo_norm: np.ndarray
    temp_profile: np.ndarray
    valid_mask: np.ndarray
    temp_cols: list[str]
    depths: np.ndarray


def is_temp_col(col: str) -> bool:
    return re.match(r"^\d+(\.\d+)?\s*m$", str(col).strip()) is not None


def parse_depth(col: str) -> float:
    return float(str(col).strip().replace(" ", "").replace("m", ""))


def load_lake_data(cfg: DataConfig) -> LakeData:
    df = pd.read_csv(cfg.csv_file)

    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # -----------------------------
    # temperature columns
    # -----------------------------
    temp_cols = [c for c in df.columns if is_temp_col(c)]
    temp_cols = sorted(temp_cols, key=parse_depth)

    if cfg.drop_first_temp_layer:
        temp_cols = temp_cols[1:]

    if len(temp_cols) == 0:
        raise ValueError("No valid temperature columns after dropping first layer.")

    depths = np.array([parse_depth(c) for c in temp_cols], dtype=np.float32)

    # -----------------------------
    # qc columns
    # -----------------------------
    qc_bad_cols = []
    for c in temp_cols:
        q = f"{c}_qc_bad"
        if q not in df.columns:
            raise ValueError(f"Missing QC column: {q}")
        qc_bad_cols.append(q)

    # -----------------------------
    # meteo
    # -----------------------------
    for c in cfg.meteo_cols:
        if c not in df.columns:
            raise ValueError(f"Missing meteo column: {c}")

    meteo_raw_base = df[list(cfg.meteo_cols)].to_numpy(dtype=np.float32)

    # -----------------------------
    # temp + valid mask
    # -----------------------------
    temp = df[temp_cols].to_numpy(dtype=np.float32)
    qc_bad = df[qc_bad_cols].to_numpy(dtype=bool)

    valid_mask = (~qc_bad) & np.isfinite(temp)
    temp_filled = np.where(np.isfinite(temp), temp, 0.0).astype(np.float32)

    # -----------------------------
    # normalize meteo
    # -----------------------------
    n = len(df)
    norm_end = min(cfg.norm_end_idx, n)
    if norm_end <= 0:
        raise ValueError("norm_end_idx must be > 0.")

    meteo_mean = meteo_raw_base[:norm_end].mean(axis=0, keepdims=True)
    meteo_std = meteo_raw_base[:norm_end].std(axis=0, keepdims=True)
    meteo_std = np.where(meteo_std < 1e-6, 1.0, meteo_std)

    meteo_norm_base = (meteo_raw_base - meteo_mean) / meteo_std

    # -----------------------------
    # optional DOY features
    # -----------------------------
    if cfg.add_doy:
        doy = df["time"].dt.dayofyear.to_numpy(dtype=np.float32)
        angle = 2.0 * np.pi * doy / cfg.doy_period
        doy_sin = np.sin(angle).astype(np.float32)[:, None]
        doy_cos = np.cos(angle).astype(np.float32)[:, None]

        meteo_norm = np.concatenate([meteo_norm_base, doy_sin, doy_cos], axis=1)
        meteo_raw = np.concatenate([meteo_raw_base, doy_sin, doy_cos], axis=1)
    else:
        meteo_norm = meteo_norm_base.astype(np.float32)
        meteo_raw = meteo_raw_base.astype(np.float32)

    return LakeData(
        time=df["time"].to_numpy(),
        meteo_raw=meteo_raw.astype(np.float32),
        meteo_norm=meteo_norm.astype(np.float32),
        temp_profile=temp_filled.astype(np.float32),
        valid_mask=valid_mask.astype(np.float32),
        temp_cols=temp_cols,
        depths=depths,
    )


def print_data_summary(data: LakeData) -> None:
    print("time length =", len(data.time))
    print("meteo_raw shape =", data.meteo_raw.shape)
    print("meteo_norm shape =", data.meteo_norm.shape)
    print("temp_profile shape =", data.temp_profile.shape)
    print("valid_mask shape =", data.valid_mask.shape)
    print("n_layers =", len(data.temp_cols))
    print("temp cols =", data.temp_cols)
    print("depths =", data.depths.tolist())

    valid_ratio = float(data.valid_mask.mean())
    print(f"overall valid ratio = {valid_ratio:.4f}")

def load_lake_data_with_given_norm(
    cfg: DataConfig,
    meteo_mean: np.ndarray,
    meteo_std: np.ndarray,
) -> LakeData:
    df = pd.read_csv(cfg.csv_file)

    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # -----------------------------
    # temperature columns
    # -----------------------------
    temp_cols = [c for c in df.columns if is_temp_col(c)]
    temp_cols = sorted(temp_cols, key=parse_depth)

    if cfg.drop_first_temp_layer:
        temp_cols = temp_cols[1:]

    if len(temp_cols) == 0:
        raise ValueError("No valid temperature columns after dropping first layer.")

    depths = np.array([parse_depth(c) for c in temp_cols], dtype=np.float32)

    # -----------------------------
    # qc columns
    # -----------------------------
    qc_bad_cols = []
    for c in temp_cols:
        q = f"{c}_qc_bad"
        if q not in df.columns:
            raise ValueError(f"Missing QC column: {q}")
        qc_bad_cols.append(q)

    # -----------------------------
    # meteo
    # -----------------------------
    for c in cfg.meteo_cols:
        if c not in df.columns:
            raise ValueError(f"Missing meteo column: {c}")

    meteo_raw_base = df[list(cfg.meteo_cols)].to_numpy(dtype=np.float32)

    # -----------------------------
    # temp + valid mask
    # -----------------------------
    temp = df[temp_cols].to_numpy(dtype=np.float32)
    qc_bad = df[qc_bad_cols].to_numpy(dtype=bool)

    valid_mask = (~qc_bad) & np.isfinite(temp)
    temp_filled = np.where(np.isfinite(temp), temp, 0.0).astype(np.float32)

    # -----------------------------
    # normalize using provided stats
    # -----------------------------
    meteo_mean = np.asarray(meteo_mean, dtype=np.float32)
    meteo_std = np.asarray(meteo_std, dtype=np.float32)
    meteo_std = np.where(meteo_std < 1e-6, 1.0, meteo_std)

    meteo_norm_base = (meteo_raw_base - meteo_mean) / meteo_std

    # -----------------------------
    # optional DOY features
    # -----------------------------
    if cfg.add_doy:
        doy = df["time"].dt.dayofyear.to_numpy(dtype=np.float32)
        angle = 2.0 * np.pi * doy / cfg.doy_period
        doy_sin = np.sin(angle).astype(np.float32)[:, None]
        doy_cos = np.cos(angle).astype(np.float32)[:, None]

        meteo_norm = np.concatenate([meteo_norm_base, doy_sin, doy_cos], axis=1)
        meteo_raw = np.concatenate([meteo_raw_base, doy_sin, doy_cos], axis=1)
    else:
        meteo_norm = meteo_norm_base.astype(np.float32)
        meteo_raw = meteo_raw_base.astype(np.float32)

    return LakeData(
        time=df["time"].to_numpy(),
        meteo_raw=meteo_raw.astype(np.float32),
        meteo_norm=meteo_norm.astype(np.float32),
        temp_profile=temp_filled.astype(np.float32),
        valid_mask=valid_mask.astype(np.float32),
        temp_cols=temp_cols,
        depths=depths,
    )