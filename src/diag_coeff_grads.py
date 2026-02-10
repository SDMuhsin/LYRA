"""Diagnostic: show per-coefficient gradient heatmap for Spectral adapter on CoLA.

Goal: determine if the DC coefficient (0,0) and low-frequency positions
dominate the gradient, creating an imbalanced optimization landscape.
"""
import sys, math, torch, torch.nn as nn
import numpy as np
sys.path.insert(0, "src")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import default_data_collator
import datasets

device = torch.device("cuda:0")

# --- Setup model with spectral adapters ---
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.float()
from spectral_adapter import get_spectral_adapter_model
model = get_spectral_adapter_model(model, ["query","key","value","dense"], p=16, q=16, scaling=1.0)
model = model.to(device)

# --- Get CoLA data (larger batch for stable gradient estimate) ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = datasets.load_dataset("glue", "cola", split="train[:128]")
def preprocess(ex):
    result = tokenizer(ex["sentence"], padding="max_length", max_length=128, truncation=True)
    result["labels"] = ex["label"]
    return result
ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
loader = DataLoader(ds, batch_size=64, collate_fn=default_data_collator)
batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items()}

# --- Forward + backward ---
model.train()
outputs = model(**batch)
loss = outputs.loss
loss.backward()

# --- Analyze per-coefficient gradient structure ---
print("=" * 70)
print("PER-COEFFICIENT GRADIENT ANALYSIS")
print("=" * 70)

# Collect all coefficient gradient matrices
all_coeffs_grads = {}
for name, param in model.named_parameters():
    if param.requires_grad and 'coeffs' in name and param.grad is not None:
        all_coeffs_grads[name] = param.grad.detach().cpu().numpy()

# Show detailed heatmap for a few representative layers
representative_layers = [
    "model.bert.encoder.layer.11.output.dense.coeffs",      # last encoder layer
    "model.bert.encoder.layer.6.output.dense.coeffs",       # mid encoder layer
    "model.bert.encoder.layer.0.attention.self.query.coeffs", # first attention
    "model.bert.encoder.layer.11.attention.self.value.coeffs", # last attention value
    "model.bert.pooler.dense.coeffs",                        # pooler (feeds classifier)
]

for layer_name in representative_layers:
    if layer_name not in all_coeffs_grads:
        print(f"\n{layer_name}: NOT FOUND")
        continue

    g = all_coeffs_grads[layer_name]  # shape (p, q) = (16, 16)
    g_abs = np.abs(g)

    print(f"\n{'='*70}")
    print(f"Layer: {layer_name}")
    print(f"  Shape: {g.shape}")
    print(f"  Frobenius norm: {np.linalg.norm(g):.6f}")
    print(f"  |DC coeff grad (0,0)|: {g_abs[0,0]:.6f}")
    print(f"  Mean |grad| (all):     {g_abs.mean():.6f}")
    print(f"  Mean |grad| (non-DC):  {g_abs[1:, 1:].mean():.6f}")
    print(f"  Max |grad| (non-DC):   {g_abs[1:, 1:].max():.6f}")
    print(f"  Ratio DC / mean(non-DC): {g_abs[0,0] / (g_abs[1:,1:].mean() + 1e-12):.1f}x")

    # Show which positions have the largest gradients
    print(f"\n  Top 10 coefficient positions by |gradient|:")
    flat_idx = np.argsort(g_abs.ravel())[::-1]
    for rank, idx in enumerate(flat_idx[:10]):
        i, j = divmod(idx, g.shape[1])
        print(f"    ({i:2d},{j:2d}): {g_abs[i,j]:.6f}  {'← DC' if (i==0 and j==0) else ''}")

    # Show gradient magnitude by frequency band
    print(f"\n  Gradient magnitude by frequency band:")
    for freq in range(min(8, g.shape[0])):
        # Elements where max(i,j) == freq  (i.e., this frequency band)
        band_vals = []
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if max(i,j) == freq:
                    band_vals.append(g_abs[i,j])
        band_mean = np.mean(band_vals)
        band_max = np.max(band_vals)
        print(f"    freq={freq:2d}: mean={band_mean:.6f}  max={band_max:.6f}  n={len(band_vals)}")

# --- Aggregate analysis across ALL layers ---
print(f"\n{'='*70}")
print("AGGREGATE ANALYSIS ACROSS ALL 73 MODULES")
print(f"{'='*70}")

dc_grads = []
nondc_grads = []
row0_grads = []
col0_grads = []
total_energy_dc = 0.0
total_energy_all = 0.0

for name, g in all_coeffs_grads.items():
    g_abs = np.abs(g)
    dc_grads.append(g_abs[0,0])
    nondc_grads.append(g_abs[1:,1:].mean())
    row0_grads.append(g_abs[0,:].mean())
    col0_grads.append(g_abs[:,0].mean())
    total_energy_dc += g[0,0]**2
    total_energy_all += (g**2).sum()

dc_grads = np.array(dc_grads)
nondc_grads = np.array(nondc_grads)

print(f"Mean |DC gradient| across layers:      {dc_grads.mean():.6f}")
print(f"Mean |non-DC gradient| across layers:   {nondc_grads.mean():.6f}")
print(f"Ratio DC / non-DC:                      {dc_grads.mean() / (nondc_grads.mean() + 1e-12):.1f}x")
print(f"DC energy fraction of total:             {total_energy_dc / (total_energy_all + 1e-12) * 100:.1f}%")
print(f"  (If uniform over 256 coefficients, expected: {100/256:.1f}%)")

# Row-0 and Column-0 analysis (mixed DC/low-freq)
print(f"\nMean |row-0 gradient| (output DC):     {np.mean(row0_grads):.6f}")
print(f"Mean |col-0 gradient| (input DC):      {np.mean(col0_grads):.6f}")
print(f"Mean |interior gradient| (i>0,j>0):    {nondc_grads.mean():.6f}")

# Compute what happens with gradient clipping
total_grad_norm = math.sqrt(sum(
    p.grad.norm().item()**2
    for p in model.parameters()
    if p.requires_grad and p.grad is not None
))
clip_factor = 1.0 / total_grad_norm  # gradient clipping at 1.0

print(f"\nTotal gradient norm: {total_grad_norm:.4f}")
print(f"Clip factor (threshold=1.0): {clip_factor:.6f}")
print(f"Post-clip DC gradient mean:     {dc_grads.mean() * clip_factor:.8f}")
print(f"Post-clip non-DC gradient mean: {nondc_grads.mean() * clip_factor:.8f}")
print(f"Post-clip ratio:                {dc_grads.mean() / (nondc_grads.mean() + 1e-12):.1f}x (unchanged by clipping)")

# What this means for Adam
print(f"\nIMPLICATION FOR ADAM OPTIMIZER:")
print(f"  Adam normalizes gradients per-element: step ≈ lr * m/(sqrt(v)+eps)")
print(f"  After many steps, step size → lr * sign(grad) when gradient is consistent")
print(f"  DC coefficient: consistent direction → fast convergence")
print(f"  High-freq coefficients: noisy direction → slow/oscillatory convergence")
print(f"  Result: DC dominates the weight update, limiting expressiveness")
