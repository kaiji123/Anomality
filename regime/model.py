"""RG-ResMoE (Ye & Borde, arXiv:2608.12251) and a size-matched plain-MLP baseline."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Block(nn.Module):
    """Two hidden layers of width `hidden`, GELU, dropout after each hidden layer."""

    def __init__(self, d_in: int, hidden: int, d_out: int = 1, dropout: float = 0.1, zero_last: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )
        if zero_last:  # experts start as "no correction"
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGResMoE(nn.Module):
    """y = base(x) + sum_k softmax(gate(x, z))_k * expert_k(x).

    Only the gate sees the regime variables z.
    """

    def __init__(self, d_x: int, d_z: int, hidden: int = 64, experts: int = 4, dropout: float = 0.1,
                 base_hidden: int = 0, gate_style: str = "block"):
        super().__init__()
        self.base = Block(d_x, base_hidden or hidden, 1, dropout)
        self.experts = nn.ModuleList([Block(d_x, hidden, 1, dropout, zero_last=True) for _ in range(experts)])
        if gate_style == "paper":  # authors' GatingNet: one hidden layer of 16, no dropout
            self.gate = nn.Sequential(nn.Linear(d_x + d_z, 16), nn.GELU(), nn.Linear(16, experts))
        else:
            self.gate = Block(d_x + d_z, hidden, experts, dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        base = self.base(x).squeeze(-1)
        r = torch.cat([e(x) for e in self.experts], dim=-1)  # [N, K]
        pi = torch.softmax(self.gate(torch.cat([x, z], dim=-1)), dim=-1)  # [N, K]
        corr = (pi * r).sum(-1)
        return base + corr, base, corr, pi

    def stage2_parameters(self):
        return list(self.experts.parameters()) + list(self.gate.parameters())


class PlainMLP(nn.Module):
    """Baseline: one Block on x only, widened to roughly match RG-ResMoE's parameter count."""

    def __init__(self, d_x: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = Block(d_x, hidden, 1, dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        return self.net(x).squeeze(-1)


def build_moe(d_x: int, d_z: int, cfg) -> RGResMoE:
    return RGResMoE(d_x, d_z, cfg.hidden, cfg.experts, cfg.dropout, cfg.base_hidden, cfg.gate_style)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def matched_width(d_x: int, target_params: int) -> int:
    """Width H so that a Block(d_x, H) has about `target_params` weights: H^2 + (d_x + 3)H + 1."""
    b = d_x + 3
    return max(8, int(round((-b + math.sqrt(b * b + 4 * (target_params - 1))) / 2)))
