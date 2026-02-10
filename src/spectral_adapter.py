"""
Truncated DCT Factored Adaptation (Spectral Adapter)

A parameter-efficient fine-tuning method that parameterizes weight updates
in the 2D DCT domain using a contiguous low-frequency coefficient block.

Key insight: Weight updates are smooth signals whose energy concentrates in
low-frequency DCT components. By restricting trainable coefficients to a
contiguous p×q low-frequency block, we enable a factored forward pass:

    delta_y = scaling * (x @ C_n[:q]^T) @ S^T @ C_m[:p]

where C_n, C_m are DCT basis matrices (frozen) and S ∈ R^{p×q} is trainable.

This achieves:
- p×q trainable parameters per module (vs LoRA's r*(m+n))
- Effective rank up to min(p,q)
- No dense ΔW reconstruction needed
- O(pq) dominant cost vs O(mn) for dense methods
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_basis(d: int, k: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute first k rows of the d-dimensional DCT-II orthonormal basis matrix.

    Returns: Tensor of shape (k, d) where row i is the i-th DCT basis vector.
    """
    n = torch.arange(d, dtype=torch.float64)
    idx = torch.arange(k, dtype=torch.float64)
    # DCT-II: C[i, j] = alpha_i * cos(pi * (2j + 1) * i / (2d))
    basis = torch.cos(torch.pi * idx[:, None] * (2 * n[None, :] + 1) / (2 * d))
    # Orthonormal scaling
    basis[0] *= 1.0 / math.sqrt(d)
    basis[1:] *= math.sqrt(2.0 / d)
    return basis.to(dtype)


class SpectralAdapterLinear(nn.Module):
    """
    A linear layer wrapped with a Truncated DCT Factored Adapter.

    Replaces: y = Wx + b
    With:     y = Wx + b + scaling * (x @ C_in^T @ S^T @ C_out)

    where C_in (q×n) and C_out (p×m) are frozen DCT basis matrices,
    and S (p×q) is the only trainable adapter parameter.
    """

    def __init__(self, base_layer: nn.Linear, p: int, q: int,
                 scaling: float = 1.0, dropout: float = 0.0,
                 d_initial: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.out_features = base_layer.out_features  # m
        self.in_features = base_layer.in_features    # n
        self.p = p
        self.q = q
        self.scaling = scaling

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Compute DCT basis matrices (frozen buffers)
        # dct_in: first q rows of n-dim DCT matrix → (q, n)
        # dct_out: first p rows of m-dim DCT matrix → (p, m)
        dtype = base_layer.weight.dtype
        self.register_buffer('dct_in', _dct_basis(self.in_features, q, dtype))
        self.register_buffer('dct_out', _dct_basis(self.out_features, p, dtype))

        # Trainable coefficient matrix: S ∈ R^{p × q}
        self.coeffs = nn.Parameter(torch.zeros(p, q, dtype=dtype))
        if d_initial > 0.0:
            # Nonzero init: small random perturbation so the adapter
            # contributes from the first forward pass (similar to VeRA's d_initial).
            nn.init.normal_(self.coeffs, mean=0, std=d_initial)
        # else: zeros → ΔW = 0 at start (identity adapter)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def get_delta_weight(self) -> torch.Tensor:
        """Reconstruct full ΔW = C_out^T @ S @ C_in (for analysis only)."""
        # dct_out: (p, m), dct_in: (q, n)
        # ΔW = dct_out^T @ coeffs @ dct_in = (m, p) @ (p, q) @ (q, n) = (m, n)
        return self.scaling * (self.dct_out.T @ self.coeffs @ self.dct_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        base_out = self.base_layer(x)

        # Spectral adapter: factored DCT computation
        # delta_y = scaling * x @ C_in^T @ S^T @ C_out
        # Step 1: project input to q-dim DCT space
        x_proj = F.linear(x, self.dct_in)       # (batch, seq, n) → (batch, seq, q)
        x_proj = self.dropout(x_proj)

        # Step 2: transform by trainable coefficients
        s_out = F.linear(x_proj, self.coeffs)    # (batch, seq, q) → (batch, seq, p)

        # Step 3: reconstruct in output space
        # Need: s_out @ dct_out = (batch, seq, p) @ (p, m) → (batch, seq, m)
        # F.linear(s_out, dct_out.T) = s_out @ dct_out.T.T = s_out @ dct_out ✓
        delta_out = F.linear(s_out, self.dct_out.t())

        return base_out + self.scaling * delta_out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"p={self.p}, q={self.q}, scaling={self.scaling}, "
                f"trainable_params={self.p * self.q}")


class SpectralAdapterModel(nn.Module):
    """
    Wrapper that applies SpectralAdapterLinear to target modules in a model.
    """

    def __init__(self, model: nn.Module, target_modules: List[str],
                 p: int = 32, q: int = 32, scaling: float = 1.0,
                 dropout: float = 0.0, d_initial: float = 0.0):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.p = p
        self.q = q
        self.scaling = scaling
        self.adapted_modules = []

        # First freeze ALL model parameters
        for param in model.parameters():
            param.requires_grad = False

        # Apply adapters (this creates trainable coeffs)
        self._apply_adapters(target_modules, p, q, scaling, dropout, d_initial)

        # Unfreeze classifier head (newly initialized, needs training)
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True

    def _apply_adapters(self, target_modules, p, q, scaling, dropout, d_initial):
        """Replace target linear layers with SpectralAdapterLinear."""
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(target in name for target in target_modules):
                continue

            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(self.model.named_modules())[parent_name]
            else:
                attr_name = parts[0]
                parent = self.model

            # Determine p, q for this layer (could be adaptive)
            layer_p = min(p, module.out_features)
            layer_q = min(q, module.in_features)

            # Replace with adapted version
            adapted = SpectralAdapterLinear(
                module, p=layer_p, q=layer_q,
                scaling=scaling, dropout=dropout,
                d_initial=d_initial
            )
            setattr(parent, attr_name, adapted)
            self.adapted_modules.append(name)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || "
              f"trainable%: {trainable / total * 100:.4f}")
        return trainable

    def get_adapter_params(self) -> int:
        """Count only adapter parameters (excluding classifier head etc.)."""
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad and 'coeffs' in name:
                count += param.numel()
        return count


def get_spectral_adapter_model(model: nn.Module,
                                target_modules: List[str],
                                p: int = 32, q: int = 32,
                                scaling: float = 1.0,
                                dropout: float = 0.0,
                                d_initial: float = 0.0) -> SpectralAdapterModel:
    """
    Apply Truncated DCT Factored Adaptation to a model.

    Args:
        model: Base model to adapt
        target_modules: List of module name patterns to adapt (e.g., ["query", "value"])
        p: Number of DCT basis vectors for output dimension
        q: Number of DCT basis vectors for input dimension
        scaling: Scaling factor for the adapter output
        dropout: Dropout probability for adapter
        d_initial: If > 0, initialize coefficients with N(0, d_initial) instead of zeros.
                   Nonzero initialization allows the adapter to contribute from the first
                   step, preventing representation drift disruption on small tasks.

    Returns:
        SpectralAdapterModel wrapping the adapted model
    """
    return SpectralAdapterModel(model, target_modules, p, q, scaling, dropout, d_initial)
