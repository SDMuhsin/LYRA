"""Diagnostic: test if nonzero coefficient initialization fixes Spectral CoLA.

Hypothesis: Zero init causes 'late adapter onset' disruption.
VeRA avoids this with d_initial=0.1.
Test: Initialize spectral coefficients with small nonzero values.
"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "src")

from spectral_adapter import SpectralAdapterLinear

# Monkey-patch the __init__ to use nonzero initialization
_original_init = SpectralAdapterLinear.__init__

D_INITIAL = None  # Set before model creation

def _patched_init(self, base_layer, p, q, scaling=1.0, dropout=0.0):
    _original_init(self, base_layer, p, q, scaling, dropout)
    if D_INITIAL is not None and D_INITIAL > 0:
        nn.init.normal_(self.coeffs, mean=0, std=D_INITIAL)

SpectralAdapterLinear.__init__ = _patched_init

# Now run training via the normal path
import subprocess, os

for d_init in [0.01, 0.1]:
    print(f"\n{'='*60}")
    print(f"Testing d_initial = {d_init}")
    print(f"{'='*60}")
