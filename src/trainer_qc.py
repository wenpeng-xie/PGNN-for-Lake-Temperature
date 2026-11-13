from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


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
    T0_profile_np: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_t_smooth: float = 1e-4,
    lambda_z_smooth: float = 1e-4,
    lambda_range: float = 1e-3,
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

    print("\n===== training =====")
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

    # load best
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # final val prediction
    pred_val, targ_val, mask_val = predict_loader(model, dl_va, device)

    np.savez_compressed(
        os.path.join(save_dir, "val_predictions_best.npz"),
        pred=pred_val,
        target=targ_val,
        mask=mask_val,
        warmup_len=warmup_len,
        target_len=target_len,
        T0_profile=T0_profile_np,
    )

    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n===== done =====")
    print("saved:", best_ckpt)
    print("saved:", os.path.join(save_dir, "val_predictions_best.npz"))
    print("saved:", os.path.join(save_dir, "history.json"))