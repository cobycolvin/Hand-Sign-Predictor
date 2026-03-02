"""MLP model definition for Week 2 experiments."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class SignMLP(nn.Module):
    """Simple configurable MLP for flattened 28x28 images."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], num_classes: int, activation: str = "relu"):
        super().__init__()
        dims: List[int] = [input_dim, *hidden_dims]

        if activation == "relu":
            act_factory = nn.ReLU
        elif activation == "tanh":
            act_factory = nn.Tanh
        elif activation == "gelu":
            act_factory = nn.GELU
        else:
            raise ValueError("activation must be one of: relu, tanh, gelu")

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_factory())
            layers.append(nn.Dropout(p=0.2))
        layers.append(nn.Linear(dims[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
