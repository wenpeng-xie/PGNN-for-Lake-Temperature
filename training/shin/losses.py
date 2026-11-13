from __future__ import annotations
import torch

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


def profile_std_torch(T: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Vertical temperature standard deviation, used as stratification-strength index.

    T, mask: (B, T, L)
    return: (B, T), unit degC
    """
    w = mask.float()
    n = w.sum(dim=-1).clamp_min(eps)

    mean = (T * w).sum(dim=-1) / n
    var = (((T - mean.unsqueeze(-1)) ** 2) * w).sum(dim=-1) / n
    return torch.sqrt(var + eps)


def profile_gmean_torch(T: torch.Tensor, mask: torch.Tensor, depths: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean absolute vertical temperature gradient.

    T, mask: (B, T, L)
    depths: (L,), unit m
    return: (B, T), unit degC/m
    """
    pair_mask = mask[:, :, 1:] * mask[:, :, :-1]

    dT = T[:, :, 1:] - T[:, :, :-1]
    dz = depths[1:] - depths[:-1]
    grad = torch.abs(dT / dz.view(1, 1, -1))

    denom = pair_mask.sum(dim=-1).clamp_min(eps)
    return (grad * pair_mask).sum(dim=-1) / denom


def normalize_by_obs(pred_x: torch.Tensor, obs_x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8):
    """
    Normalize both pred and obs using observed mean/std within valid samples.
    This makes STD/G/H losses comparable in scale.
    """
    w = valid_mask.float()
    n = w.sum().clamp_min(eps)

    obs_mean = (obs_x * w).sum() / n
    obs_var = (((obs_x - obs_mean) ** 2) * w).sum() / n
    obs_std = torch.sqrt(obs_var + eps)

    return (pred_x - obs_mean) / obs_std, (obs_x - obs_mean) / obs_std

