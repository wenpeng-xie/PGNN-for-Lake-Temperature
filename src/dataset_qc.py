from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WarmupProfileDatasetQC(Dataset):
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

        # 统一初始状态：始终使用训练集平均温度剖面
        T0 = self.T0_profile

        return (
            torch.from_numpy(T0),
            torch.from_numpy(x_norm),
            torch.from_numpy(x_raw),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )