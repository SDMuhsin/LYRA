"""
Phase 3 Supplementary: Compare DCT vs DFT energy capture at PEFT-relevant parameter counts.

Re-trains BERT briefly, computes ΔW, then for each adapted weight matrix:
- Computes energy captured by top-k DCT vs DFT coefficients for k = 50, 100, 250, 500, 1000, 2000
- Also computes SVD energy at equivalent parameter counts
- This answers: does DCT's advantage hold at the sparse regime used by PEFT?
"""
import gc
import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

SEED = 42
RESULTS_DIR = "./results/phase3_diagnostic"
K_VALUES = [50, 100, 250, 500, 1000, 2000, 5000]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_delta_weights(model_name="bert-base-uncased", task_name="mrpc",
                      num_epochs=3, lr=2e-5, batch_size=32) -> Tuple[Dict[str, torch.Tensor], float]:
    """Train and return ΔW for all 2D weight matrices."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(model_name, num_labels=2, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    pretrained_weights = {name: param.data.clone().cpu()
                          for name, param in model.named_parameters() if param.dim() == 2}
    model.to(device)

    raw_datasets = load_dataset("glue", task_name)
    def preprocess(examples):
        result = tokenizer(examples["sentence1"], examples["sentence2"],
                           padding=False, max_length=128, truncation=True)
        result["labels"] = examples["label"]
        return result

    processed = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)
    train_loader = DataLoader(processed["train"], shuffle=True,
                              collate_fn=DataCollatorWithPadding(tokenizer), batch_size=batch_size)
    eval_loader = DataLoader(processed["validation"], shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer), batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    metric = evaluate.load("glue", task_name)

    best_f1 = 0.0
    best_weights = None
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                preds = model(**batch).logits.argmax(dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
        result = metric.compute()
        f1 = result.get("f1", 0.0)
        logger.info(f"Epoch {epoch+1}: F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_weights = {name: param.data.clone().cpu()
                            for name, param in model.named_parameters() if param.dim() == 2}

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # Compute ΔW
    delta_weights = {}
    for name in pretrained_weights:
        if name in best_weights:
            dW = best_weights[name] - pretrained_weights[name]
            if dW.shape[0] >= 2 and dW.shape[1] >= 2:
                delta_weights[name] = dW.float()

    return delta_weights, best_f1


def dct2_torch(x):
    """Compute 2D Type-II DCT using scipy for accuracy."""
    try:
        from scipy.fft import dctn
        return torch.from_numpy(dctn(x.numpy(), type=2, norm='ortho'))
    except ImportError:
        # Fallback: approximate DCT via FFT
        m, n = x.shape
        # Row DCT
        xr = torch.cat([x, x.flip(dims=[1])], dim=1)
        cr = torch.fft.rfft(xr, dim=1)[:, :n].real
        # Column DCT
        xc = torch.cat([cr, cr.flip(dims=[0])], dim=0)
        cc = torch.fft.rfft(xc, dim=0)[:m, :].real
        return cc


def analyze_energy_at_k(delta_weights: Dict[str, torch.Tensor]) -> List[Dict]:
    """For each ΔW, compute energy captured at various k values for DFT, DCT, and SVD."""
    results = []

    for name, dW in delta_weights.items():
        m, n = dW.shape
        total_entries = m * n
        total_energy = (dW ** 2).sum().item()
        if total_energy < 1e-20:
            continue

        result = {"name": name, "shape": [m, n], "total_energy": total_energy}

        # --- DFT: sort magnitudes, compute cumulative energy ---
        dft = torch.fft.fft2(dW)
        dft_mag2 = (dft.abs() ** 2).flatten()
        dft_sorted, _ = torch.sort(dft_mag2, descending=True)
        dft_cumsum = torch.cumsum(dft_sorted, dim=0)
        # Normalize: DFT energy relates to spatial energy by Parseval's theorem
        dft_total = dft_cumsum[-1].item()

        # --- DCT: sort magnitudes, compute cumulative energy ---
        dct = dct2_torch(dW)
        dct_mag2 = (dct.abs() ** 2).flatten()
        dct_sorted, _ = torch.sort(dct_mag2, descending=True)
        dct_cumsum = torch.cumsum(dct_sorted, dim=0)
        dct_total = dct_cumsum[-1].item()

        # --- SVD ---
        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        sv_energy = S ** 2
        sv_cumsum = torch.cumsum(sv_energy, dim=0)
        sv_total = sv_cumsum[-1].item()

        for k in K_VALUES:
            # DFT energy at k coefficients
            if k <= len(dft_sorted):
                dft_energy_k = dft_cumsum[k - 1].item() / dft_total
            else:
                dft_energy_k = 1.0
            result[f"dft_energy_at_k{k}"] = dft_energy_k

            # DCT energy at k coefficients
            if k <= len(dct_sorted):
                dct_energy_k = dct_cumsum[k - 1].item() / dct_total
            else:
                dct_energy_k = 1.0
            result[f"dct_energy_at_k{k}"] = dct_energy_k

            # SVD energy at equivalent parameter budget
            # k parameters with SVD: rank r needs r*(m+n) params
            # So for budget k: r = k // (m+n)
            r = k // (m + n)
            if r >= 1 and r <= len(S):
                svd_energy_k = sv_cumsum[r - 1].item() / sv_total
            elif r >= len(S):
                svd_energy_k = 1.0
            else:
                svd_energy_k = 0.0  # Can't even do rank-1
            result[f"svd_energy_at_k{k}"] = svd_energy_k
            result[f"svd_rank_at_k{k}"] = r

        # Also compute: DCT advantage ratio over DFT at each k
        for k in K_VALUES:
            dft_e = result[f"dft_energy_at_k{k}"]
            dct_e = result[f"dct_energy_at_k{k}"]
            result[f"dct_over_dft_ratio_k{k}"] = dct_e / dft_e if dft_e > 0 else float('inf')

        results.append(result)

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("Phase 3 Supplementary: Low-k Energy Comparison")

    # Get ΔW (reuses training from previous analysis)
    delta_weights, best_f1 = get_delta_weights()
    logger.info(f"Training F1: {best_f1:.4f}")

    # Analyze
    results = analyze_energy_at_k(delta_weights)

    # Save
    output_path = os.path.join(RESULTS_DIR, "low_k_energy_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    # Focus on transformer encoder layers
    encoder_results = [r for r in results if "encoder" in r["name"]]

    print("\n" + "=" * 120)
    print("ENERGY CAPTURED AT LOW k (encoder layers, mean across all)")
    print("=" * 120)

    # Group by layer type
    types = {
        "attn_qkv": [r for r in encoder_results if "attention.self" in r["name"]],
        "attn_out": [r for r in encoder_results if "attention.output" in r["name"]],
        "ffn_inter": [r for r in encoder_results if "intermediate" in r["name"]],
        "ffn_out": [r for r in encoder_results if "output.dense" in r["name"] and "attention" not in r["name"]],
    }

    for type_name, type_results in types.items():
        if not type_results:
            continue
        print(f"\n--- {type_name} ({len(type_results)} matrices) ---")
        print(f"{'k':>6} | {'DFT energy%':>12} | {'DCT energy%':>12} | {'SVD energy%':>12} | {'DCT/DFT ratio':>14} | {'SVD rank':>8}")
        for k in K_VALUES:
            dft_vals = [r[f"dft_energy_at_k{k}"] for r in type_results]
            dct_vals = [r[f"dct_energy_at_k{k}"] for r in type_results]
            svd_vals = [r[f"svd_energy_at_k{k}"] for r in type_results]
            ratio_vals = [r[f"dct_over_dft_ratio_k{k}"] for r in type_results]
            rank_vals = [r[f"svd_rank_at_k{k}"] for r in type_results]
            print(f"{k:>6} | {np.mean(dft_vals)*100:>11.2f}% | {np.mean(dct_vals)*100:>11.2f}% | "
                  f"{np.mean(svd_vals)*100:>11.2f}% | {np.mean(ratio_vals):>14.2f}x | {np.mean(rank_vals):>8.1f}")

    # Layer-by-layer comparison at k=1000
    print("\n" + "=" * 120)
    print("PER-LAYER COMPARISON AT k=1000")
    print("=" * 120)
    print(f"{'name':<55} {'DFT%':>8} {'DCT%':>8} {'SVD%':>8} {'DCT/DFT':>8}")
    for r in results:
        if "encoder" in r["name"] or "pooler" in r["name"] or "classifier" in r["name"]:
            name_short = r["name"].replace("bert.encoder.layer.", "L").replace(".attention.self.", ".attn.").replace(".weight", "")
            dft_e = r["dft_energy_at_k1000"] * 100
            dct_e = r["dct_energy_at_k1000"] * 100
            svd_e = r["svd_energy_at_k1000"] * 100
            ratio = r["dct_over_dft_ratio_k1000"]
            print(f"{name_short:<55} {dft_e:>7.2f}% {dct_e:>7.2f}% {svd_e:>7.2f}% {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
