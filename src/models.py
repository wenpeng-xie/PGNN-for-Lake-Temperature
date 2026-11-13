from __future__ import annotations

import torch
import torch.nn as nn


class DeltaTUpdate(nn.Module):
    """
    dT = f(T_cur, x_t)
    gate = sigmoid(g(T_cur, x_t))
    T_next = T_cur + dt_scale * gate * dT
    """
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim: int = 64,
        dt_scale: float = 0.2,
    ):
        super().__init__()
        self.dt_scale = dt_scale

        in_dim = n_layers + input_dim

        self.delta_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_layers),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_layers),
            nn.Sigmoid(),
        )

    def forward(self, T_cur, x_t):
        inp = torch.cat([T_cur, x_t], dim=-1)
        dT = self.delta_net(inp)
        gate = self.gate_net(inp)
        T_next = T_cur + self.dt_scale * gate * dT
        return T_next, dT, gate


class ResidualSSM(nn.Module):
    """
    T-based SSM:
      T_{t+1} = T_t + dt_scale * gate(T_t, x_t) * dT(T_t, x_t)

    这里保留类名 ResidualSSM，
    这样你原来的 import 和训练主脚本基本不用改。
    """
    def __init__(
        self,
        input_dim: int,
        n_layers: int = 11,
        hidden_dim: int = 64,
        update_scale: float = 0.2,   # 继续沿用原参数名，内部当 dt_scale 用
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.state_update = DeltaTUpdate(
            input_dim=input_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dt_scale=update_scale,
        )

    def forward(self, T0, x_seq, x_raw_seq=None, return_diagnostics: bool = False):
        """
        T0    : (B, L)
        x_seq : (B, T, D)
        """
        B, T, D = x_seq.shape
        assert D == self.input_dim

        T_cur = T0
        preds = []

        dT_norms = []
        gate_means = []

        for t in range(T):
            x_t = x_seq[:, t, :]
            T_cur, dT, gate = self.state_update(T_cur, x_t)
            preds.append(T_cur.unsqueeze(1))

            if return_diagnostics:
                dT_norms.append(dT.norm(dim=-1).mean().detach())
                gate_means.append(gate.mean().detach())

        T_pred = torch.cat(preds, dim=1)

        if not return_diagnostics:
            return T_pred

        diagnostics = {
            "dT_norm_mean": torch.stack(dT_norms).mean().item(),
            "gate_mean": torch.stack(gate_means).mean().item(),
        }
        return T_pred, diagnostics