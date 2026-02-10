"""
DyLoRA: Dynamic Low-Rank Adaptation

Trains LoRA at multiple ranks simultaneously by randomly sampling
the active rank at each forward pass during training. At each step,
a rank b is sampled from {1, ..., r_max} and only the first b columns
of A and rows of B are used. At inference, the full rank is used.

This enables search-free rank selection: a single training run produces
adapters that work across all ranks from 1 to r_max.

Reference:
    Valipour et al., "DyLoRA: Parameter Efficient Tuning of Pre-trained
    Models using Dynamic Search-Free Low-Rank Adaptation", EACL 2023.
    https://arxiv.org/abs/2210.07558
"""
import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


class DyLoRALinear(nn.Module):
    """
    Linear layer with DyLoRA adaptation.

    During training: samples a random rank b in {1, ..., r} and uses
        A[:, :b] @ B[:b, :] with scaling alpha/r.
    During eval: uses full rank A @ B with scaling alpha/r.
    """

    def __init__(self, base_layer: nn.Linear, r: int,
                 alpha: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        if isinstance(base_layer, Conv1D):
            self.in_features = base_layer.nx
            self.out_features = base_layer.nf
        else:
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        dtype = base_layer.weight.dtype

        # LoRA A: (r, in_features) — same layout as standard LoRA
        # Initialized with Kaiming uniform (same as LoRA reference impl)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA B: (out_features, r) — initialized to zero (no-op at init)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype))

        # Dropout on input
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if self.training:
            # Sample random rank for this forward pass
            b = random.randint(1, self.r)
            # Truncated LoRA: use first b components
            # A_trunc: (b, in_features), B_trunc: (out_features, b)
            lora_out = F.linear(F.linear(self.dropout(x), self.lora_A[:b, :]),
                                self.lora_B[:, :b])
        else:
            # Full rank at inference
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)

        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}, "
                f"trainable_params={self.r * self.in_features + self.out_features * self.r}")


class DyLoRAModel(nn.Module):
    """
    Wrapper that applies DyLoRA adapters to target modules in a model.
    Follows the same pattern as SpectralAdapterModel.
    """

    def __init__(self, model: nn.Module, target_modules: List[str],
                 r: int = 8, alpha: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.r = r
        self.adapted_modules = []

        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Apply DyLoRA adapters
        self._apply_adapters(target_modules, r, alpha, dropout)

        # Unfreeze classifier head (needs training)
        for name, param in model.named_parameters():
            if 'classifier' in name or 'score' in name:
                param.requires_grad = True

    def _apply_adapters(self, target_modules, r, alpha, dropout):
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, (nn.Linear, Conv1D)):
                continue
            if not any(target in name for target in target_modules):
                continue

            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(self.model.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = self.model

            adapted = DyLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, adapted)
            self.adapted_modules.append(name)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {trainable / total * 100:.4f}")
        return trainable


def get_dylora_model(model: nn.Module,
                     target_modules: List[str],
                     r: int = 8,
                     alpha: Optional[int] = None,
                     dropout: float = 0.0) -> DyLoRAModel:
    """
    Apply DyLoRA adapters to a model.

    Args:
        model: Base model to adapt
        target_modules: List of module name patterns to adapt
        r: Maximum LoRA rank (during training, ranks 1..r are sampled)
        alpha: LoRA alpha scaling (defaults to r)
        dropout: Dropout probability

    Returns:
        DyLoRAModel wrapping the adapted model
    """
    return DyLoRAModel(model, target_modules, r, alpha, dropout)
