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
from transformers.pytorch_utils import Conv1D


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


def _generate_freq_indices(d: int, k: int, mode: str = "contiguous",
                           exponent: float = 2.0) -> List[int]:
    """
    Generate k frequency indices in [0, d-1] according to the chosen strategy.

    Args:
        d: Full dimension (e.g., 768).
        k: Number of frequencies to select.
        mode: "contiguous" → [0, ..., k-1] (original behaviour).
              "geometric"  → power-spaced indices over [0, d//2],
                              giving dense low-freq and sparse mid/high-freq.
              "hybrid"     → first 3k/4 contiguous low-freq, remaining k/4
                              geometrically spread over [k, d//2].
              "geometric_half" → power-spaced indices over [0, d//4],
                              more conservative coverage than geometric.
        exponent: Power for geometric spacing (default 2.0 = quadratic).
                  1.0 = linear (uniform), 3.0 = cubic (denser low-freq).

    Returns:
        Sorted list of k unique integer indices.
    """
    if mode == "contiguous":
        return list(range(k))
    elif mode == "geometric":
        half = d // 2
        raw = [round(half * (i / (k - 1)) ** exponent) for i in range(k)]
        # Remove collisions while preserving order
        seen: set = set()
        unique: List[int] = []
        for v in raw:
            while v in seen:
                v += 1
            seen.add(v)
            unique.append(v)
        return sorted(unique)
    elif mode == "geometric_half":
        quarter = d // 4
        raw = [round(quarter * (i / (k - 1)) ** exponent) for i in range(k)]
        seen: set = set()
        unique: List[int] = []
        for v in raw:
            while v in seen:
                v += 1
            seen.add(v)
            unique.append(v)
        return sorted(unique)
    elif mode == "hybrid":
        # Dense low-frequency block + geometrically spread high-frequency probes
        n_low = (3 * k) // 4          # e.g., 12 of 16
        n_high = k - n_low            # e.g., 4 of 16
        low = list(range(n_low))
        half = d // 2
        # Spread n_high probes over [n_low, half]
        span = half - n_low
        high = [n_low + round(span * ((i + 1) / n_high) ** exponent) for i in range(n_high)]
        # Deduplicate
        seen: set = set(low)
        unique_high: List[int] = []
        for v in high:
            while v in seen:
                v += 1
            seen.add(v)
            unique_high.append(v)
        return sorted(low + unique_high)
    else:
        raise ValueError(f"Unknown freq_mode: {mode!r}. Choose 'contiguous', 'geometric', 'geometric_half', or 'hybrid'.")


def _dct_basis_at_indices(d: int, freq_indices: List[int],
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute DCT-II rows at *arbitrary* frequency indices with orthonormal scaling.

    Args:
        d: Full dimension.
        freq_indices: Which DCT rows to materialise (length k).
        dtype: Output dtype.

    Returns:
        Tensor of shape (k, d).
    """
    k = len(freq_indices)
    n = torch.arange(d, dtype=torch.float64)
    idx = torch.tensor(freq_indices, dtype=torch.float64)
    # DCT-II: C[i, j] = alpha_i * cos(pi * i * (2j + 1) / (2d))
    basis = torch.cos(torch.pi * idx[:, None] * (2 * n[None, :] + 1) / (2 * d))
    # Orthonormal scaling
    for r in range(k):
        if freq_indices[r] == 0:
            basis[r] *= 1.0 / math.sqrt(d)
        else:
            basis[r] *= math.sqrt(2.0 / d)
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
                 d_initial: float = 0.0, freq_mode: str = "contiguous",
                 freq_exponent: float = 2.0, factored_rank: int = 0,
                 learn_scaling: bool = False):
        super().__init__()
        self.base_layer = base_layer
        if isinstance(base_layer, Conv1D):
            self.out_features = base_layer.nf
            self.in_features = base_layer.nx
        else:
            self.out_features = base_layer.out_features
            self.in_features = base_layer.in_features
        self.p = p
        self.q = q
        self.factored_rank = factored_rank
        self.learn_scaling = learn_scaling

        # Scaling: learnable per-module scalar or fixed constant
        if learn_scaling:
            self.log_scaling = nn.Parameter(
                torch.tensor(math.log(max(scaling, 1e-6)),
                             dtype=base_layer.weight.dtype))
        else:
            self.scaling = scaling

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Compute DCT basis matrices (frozen buffers)
        # dct_in: selected q rows of n-dim DCT matrix → (q, n)
        # dct_out: selected p rows of m-dim DCT matrix → (p, m)
        dtype = base_layer.weight.dtype
        if freq_mode == "contiguous":
            self.register_buffer('dct_in', _dct_basis(self.in_features, q, dtype))
            self.register_buffer('dct_out', _dct_basis(self.out_features, p, dtype))
        else:
            freq_in = _generate_freq_indices(self.in_features, q, freq_mode, freq_exponent)
            freq_out = _generate_freq_indices(self.out_features, p, freq_mode, freq_exponent)
            self.register_buffer('dct_in', _dct_basis_at_indices(self.in_features, freq_in, dtype))
            self.register_buffer('dct_out', _dct_basis_at_indices(self.out_features, freq_out, dtype))

        # Trainable coefficient matrix
        if factored_rank > 0:
            # Factored: S = A @ B, where A ∈ R^{p × r}, B ∈ R^{r × q}
            # Params per module = p*r + r*q instead of p*q
            self.coeffs_A = nn.Parameter(torch.zeros(p, factored_rank, dtype=dtype))
            self.coeffs_B = nn.Parameter(torch.zeros(factored_rank, q, dtype=dtype))
            if d_initial > 0.0:
                nn.init.normal_(self.coeffs_A, mean=0, std=d_initial)
                nn.init.normal_(self.coeffs_B, mean=0, std=d_initial)
        else:
            # Dense: S ∈ R^{p × q}
            self.coeffs = nn.Parameter(torch.zeros(p, q, dtype=dtype))
            if d_initial > 0.0:
                nn.init.normal_(self.coeffs, mean=0, std=d_initial)
        # else: zeros → ΔW = 0 at start (identity adapter)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _get_scaling(self) -> float:
        """Return the effective scaling factor (learnable or fixed)."""
        if self.learn_scaling:
            return torch.exp(self.log_scaling)
        return self.scaling

    def _get_S(self) -> torch.Tensor:
        """Return the effective S matrix (factored or dense)."""
        if self.factored_rank > 0:
            return self.coeffs_A @ self.coeffs_B
        return self.coeffs

    def get_delta_weight(self) -> torch.Tensor:
        """Reconstruct full ΔW = C_out^T @ S @ C_in (for analysis only)."""
        S = self._get_S()
        return self._get_scaling() * (self.dct_out.T @ S @ self.dct_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        base_out = self.base_layer(x)

        # Spectral adapter: factored DCT computation
        # Step 1: project input to q-dim DCT space
        x_proj = F.linear(x, self.dct_in)       # (batch, seq, n) → (batch, seq, q)
        x_proj = self.dropout(x_proj)

        # Step 2: transform by trainable coefficients
        if self.factored_rank > 0:
            # Factored S = A @ B: two smaller matmuls instead of one
            x_mid = F.linear(x_proj, self.coeffs_B)   # (batch, seq, q) → (batch, seq, r)
            s_out = F.linear(x_mid, self.coeffs_A)     # (batch, seq, r) → (batch, seq, p)
        else:
            s_out = F.linear(x_proj, self.coeffs)      # (batch, seq, q) → (batch, seq, p)

        # Step 3: reconstruct in output space
        delta_out = F.linear(s_out, self.dct_out.t())

        return base_out + self._get_scaling() * delta_out

    def extra_repr(self) -> str:
        scaling_str = f"learn_scaling=True" if self.learn_scaling else f"scaling={self.scaling}"
        if self.factored_rank > 0:
            params = self.p * self.factored_rank + self.factored_rank * self.q
            return (f"in_features={self.in_features}, out_features={self.out_features}, "
                    f"p={self.p}, q={self.q}, factored_rank={self.factored_rank}, "
                    f"{scaling_str}, trainable_params={params}")
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"p={self.p}, q={self.q}, {scaling_str}, "
                f"trainable_params={self.p * self.q}")


class SpectralAdapterModel(nn.Module):
    """
    Wrapper that applies SpectralAdapterLinear to target modules in a model.
    """

    def __init__(self, model: nn.Module, target_modules: List[str],
                 p: int = 32, q: int = 32, scaling: float = 1.0,
                 dropout: float = 0.0, d_initial: float = 0.0,
                 freq_mode: str = "contiguous", freq_exponent: float = 2.0,
                 factored_rank: int = 0, learn_scaling: bool = False):
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
        self._apply_adapters(target_modules, p, q, scaling, dropout, d_initial, freq_mode,
                             freq_exponent, factored_rank, learn_scaling)

        # Unfreeze classifier head (newly initialized, needs training)
        for name, param in model.named_parameters():
            if 'classifier' in name or 'score' in name:
                param.requires_grad = True

    def _apply_adapters(self, target_modules, p, q, scaling, dropout, d_initial,
                        freq_mode="contiguous", freq_exponent=2.0, factored_rank=0,
                        learn_scaling=False):
        """Replace target linear layers with SpectralAdapterLinear."""
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, (nn.Linear, Conv1D)):
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
            if isinstance(module, Conv1D):
                out_f, in_f = module.nf, module.nx
            else:
                out_f, in_f = module.out_features, module.in_features
            layer_p = min(p, out_f)
            layer_q = min(q, in_f)

            # Replace with adapted version
            adapted = SpectralAdapterLinear(
                module, p=layer_p, q=layer_q,
                scaling=scaling, dropout=dropout,
                d_initial=d_initial, freq_mode=freq_mode,
                freq_exponent=freq_exponent,
                factored_rank=factored_rank,
                learn_scaling=learn_scaling,
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
            if param.requires_grad and ('coeffs' in name or 'log_scaling' in name):
                count += param.numel()
        return count


def get_spectral_adapter_model(model: nn.Module,
                                target_modules: List[str],
                                p: int = 32, q: int = 32,
                                scaling: float = 1.0,
                                dropout: float = 0.0,
                                d_initial: float = 0.0,
                                freq_mode: str = "contiguous",
                                freq_exponent: float = 2.0,
                                factored_rank: int = 0,
                                learn_scaling: bool = False) -> SpectralAdapterModel:
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
        freq_mode: Frequency selection strategy. "contiguous" uses [0..k-1] (default).
                   "geometric" uses power-spaced indices over [0, d//2].
        freq_exponent: Power for geometric spacing (default 2.0). 1.0=linear, 3.0=cubic.
        factored_rank: If > 0, factor S = A(p,r) @ B(r,q) for wider frequency
                       coverage at same param count (r=factored_rank). 0 = dense S.
        learn_scaling: If True, each adapter module gets a learnable log-space scaling
                       parameter initialized to log(scaling). Adds 1 param per module.

    Returns:
        SpectralAdapterModel wrapping the adapted model
    """
    return SpectralAdapterModel(model, target_modules, p, q, scaling, dropout, d_initial,
                                freq_mode, freq_exponent, factored_rank, learn_scaling)
