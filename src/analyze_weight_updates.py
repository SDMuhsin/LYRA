"""
Phase 3 Diagnostic: Characterize the structural properties of transformer weight updates.

Fine-tunes BERT-base on MRPC (full fine-tuning, single seed), then computes ΔW = W_finetuned - W_pretrained
for every weight matrix and analyzes:
1. SVD spectrum (rank structure)
2. 2D DFT energy distribution (spectral sparsity)
3. Spatial correlation patterns
4. Row/column independence (separability)

The goal is to identify which signal processing principle best describes ΔW structure,
to guide design of a novel PEFT method from first principles.
"""
import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import evaluate
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Disable TF32 for reproducibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

TASK_TO_KEYS = {
    "mrpc": ("sentence1", "sentence2"),
}
SEED = 42
RESULTS_DIR = "./results/phase3_diagnostic"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_full_finetune(model_name: str, task_name: str, num_epochs: int = 3,
                        lr: float = 2e-5, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
    """
    Full fine-tuning of BERT on MRPC. Returns pretrained weights, finetuned weights, and eval metrics.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    config = AutoConfig.from_pretrained(model_name, num_labels=2, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Save pretrained weights (deep copy of all weight matrices)
    pretrained_weights = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:  # Only 2D weight matrices
            pretrained_weights[name] = param.data.clone().cpu()

    model.to(device)

    # Load data
    raw_datasets = load_dataset("glue", task_name)
    s1_key, s2_key = TASK_TO_KEYS[task_name]

    def preprocess(examples):
        texts = (examples[s1_key], examples[s2_key])
        result = tokenizer(*texts, padding=False, max_length=128, truncation=True)
        result["labels"] = examples["label"]
        return result

    processed = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)
    train_loader = DataLoader(processed["train"], shuffle=True,
                              collate_fn=DataCollatorWithPadding(tokenizer),
                              batch_size=batch_size)
    eval_loader = DataLoader(processed["validation"], shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer),
                             batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=0, num_training_steps=total_steps)

    metric = evaluate.load("glue", task_name)

    # Training loop
    logger.info(f"Training full fine-tuning: {num_epochs} epochs, lr={lr}, batch={batch_size}")
    best_f1 = 0.0
    best_weights = None

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluate
        model.eval()
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())

        eval_result = metric.compute()
        f1 = eval_result.get("f1", 0.0)
        logger.info(f"Epoch {epoch+1}: F1={f1:.4f}, Acc={eval_result.get('accuracy', 0.0):.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_weights = {}
            for name, param in model.named_parameters():
                if param.dim() == 2:
                    best_weights[name] = param.data.clone().cpu()

    logger.info(f"Best F1: {best_f1:.4f}")
    return pretrained_weights, best_weights, {"best_f1": best_f1}


def analyze_delta_w(pretrained: Dict[str, torch.Tensor],
                    finetuned: Dict[str, torch.Tensor]) -> List[Dict]:
    """
    For each weight matrix, compute ΔW and analyze its structural properties.
    """
    results = []

    for name in pretrained:
        if name not in finetuned:
            continue

        W0 = pretrained[name].float()
        W1 = finetuned[name].float()
        dW = W1 - W0

        m, n = dW.shape
        if m < 2 or n < 2:
            continue

        result = {"name": name, "shape": list(dW.shape)}

        # 1. SVD Analysis: rank structure
        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        total_energy = (S ** 2).sum().item()
        if total_energy < 1e-20:
            # No meaningful update for this matrix
            result["svd_total_energy"] = 0.0
            result["skipped"] = True
            results.append(result)
            continue

        cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
        # Effective rank: number of singular values for 90%, 95%, 99% energy
        rank_90 = (cumulative_energy < 0.90).sum().item() + 1
        rank_95 = (cumulative_energy < 0.95).sum().item() + 1
        rank_99 = (cumulative_energy < 0.99).sum().item() + 1
        max_rank = min(m, n)

        result["svd_total_energy"] = total_energy
        result["svd_rank_90"] = int(rank_90)
        result["svd_rank_95"] = int(rank_95)
        result["svd_rank_99"] = int(rank_99)
        result["svd_max_rank"] = int(max_rank)
        result["svd_rank_90_frac"] = rank_90 / max_rank
        result["svd_top1_energy_frac"] = (S[0] ** 2).item() / total_energy
        result["svd_top5_energy_frac"] = (S[:5] ** 2).sum().item() / total_energy
        result["svd_top10_energy_frac"] = (S[:min(10, len(S))] ** 2).sum().item() / total_energy

        # Parameters needed for rank-k SVD approximation to capture 90% energy
        # rank_90 components need rank_90 * (m + n) parameters
        result["svd_params_90"] = int(rank_90 * (m + n))

        # 2. DFT Analysis: spectral sparsity
        dW_fft = torch.fft.fft2(dW)
        fft_magnitudes = torch.abs(dW_fft).flatten()
        fft_energy = (fft_magnitudes ** 2)
        fft_total_energy = fft_energy.sum().item()
        fft_sorted, _ = torch.sort(fft_energy, descending=True)
        fft_cumulative = torch.cumsum(fft_sorted, dim=0) / fft_total_energy

        fft_k_90 = (fft_cumulative < 0.90).sum().item() + 1
        fft_k_95 = (fft_cumulative < 0.95).sum().item() + 1
        fft_k_99 = (fft_cumulative < 0.99).sum().item() + 1
        total_coeffs = m * n

        result["fft_k_90"] = int(fft_k_90)
        result["fft_k_95"] = int(fft_k_95)
        result["fft_k_99"] = int(fft_k_99)
        result["fft_total_coeffs"] = int(total_coeffs)
        result["fft_k_90_frac"] = fft_k_90 / total_coeffs
        # Parameters needed: fft_k_90 real-valued coefficients
        result["fft_params_90"] = int(fft_k_90)

        # 3. DCT Analysis (via real FFT, approximation)
        # Use Type-II DCT via scipy-like approach with torch
        # DCT-II of each row, then DCT-II of each column
        # Approximate with real part of FFT of mirrored signal
        dW_mirror_rows = torch.cat([dW, dW.flip(dims=[1])], dim=1)
        dct_rows = torch.fft.rfft(dW_mirror_rows, dim=1)[:, :n].real
        dW_mirror_cols = torch.cat([dct_rows, dct_rows.flip(dims=[0])], dim=0)
        dct_2d = torch.fft.rfft(dW_mirror_cols, dim=0)[:m, :].real

        dct_magnitudes = torch.abs(dct_2d).flatten()
        dct_energy = dct_magnitudes ** 2
        dct_total_energy = dct_energy.sum().item()
        if dct_total_energy > 1e-20:
            dct_sorted, _ = torch.sort(dct_energy, descending=True)
            dct_cumulative = torch.cumsum(dct_sorted, dim=0) / dct_total_energy
            dct_k_90 = (dct_cumulative < 0.90).sum().item() + 1
            dct_k_95 = (dct_cumulative < 0.95).sum().item() + 1
            result["dct_k_90"] = int(dct_k_90)
            result["dct_k_95"] = int(dct_k_95)
            result["dct_k_90_frac"] = dct_k_90 / total_coeffs
            result["dct_params_90"] = int(dct_k_90)

        # 4. Row/Column correlation analysis (separability)
        # If ΔW ≈ u * v^T (rank-1), rows are highly correlated
        # Compute correlation between consecutive rows
        if m > 1:
            row_corrs = []
            for i in range(min(m - 1, 50)):  # Sample up to 50 pairs
                r1 = dW[i].float()
                r2 = dW[i + 1].float()
                if r1.norm() > 1e-10 and r2.norm() > 1e-10:
                    corr = torch.dot(r1, r2) / (r1.norm() * r2.norm())
                    row_corrs.append(abs(corr.item()))
            result["row_mean_abs_corr"] = float(np.mean(row_corrs)) if row_corrs else 0.0

        # 5. Frobenius norm and relative magnitude
        result["frobenius_norm"] = dW.norm().item()
        result["relative_norm"] = dW.norm().item() / W0.norm().item() if W0.norm().item() > 0 else 0.0

        # 6. Sparsity (fraction of near-zero elements)
        threshold = 0.01 * dW.abs().max().item() if dW.abs().max().item() > 0 else 1e-10
        result["spatial_sparsity"] = (dW.abs() < threshold).float().mean().item()

        results.append(result)

    return results


def summarize_results(results: List[Dict]) -> Dict:
    """
    Aggregate results across layers and produce summary statistics.
    """
    # Filter out skipped matrices
    active = [r for r in results if not r.get("skipped", False)]

    if not active:
        return {"error": "No active weight matrices found"}

    summary = {
        "total_matrices": len(active),
        "total_skipped": len(results) - len(active),
    }

    # Categorize matrices
    categories = {
        "attention_qkv": [],
        "attention_output": [],
        "ffn_intermediate": [],
        "ffn_output": [],
        "embeddings": [],
        "classifier": [],
        "pooler": [],
        "other": [],
    }

    for r in active:
        name = r["name"]
        if "attention.self" in name and any(k in name for k in ["query", "key", "value"]):
            categories["attention_qkv"].append(r)
        elif "attention.output" in name:
            categories["attention_output"].append(r)
        elif "intermediate" in name:
            categories["ffn_intermediate"].append(r)
        elif "output.dense" in name and "attention" not in name:
            categories["ffn_output"].append(r)
        elif "embedding" in name:
            categories["embeddings"].append(r)
        elif "classifier" in name:
            categories["classifier"].append(r)
        elif "pooler" in name:
            categories["pooler"].append(r)
        else:
            categories["other"].append(r)

    # Per-category summaries
    for cat_name, cat_results in categories.items():
        if not cat_results:
            continue
        n = len(cat_results)
        summary[f"{cat_name}_count"] = n

        for metric in ["svd_rank_90", "svd_rank_90_frac", "svd_top1_energy_frac",
                        "svd_top10_energy_frac", "svd_params_90",
                        "fft_k_90", "fft_k_90_frac", "fft_params_90",
                        "relative_norm", "row_mean_abs_corr", "spatial_sparsity"]:
            vals = [r[metric] for r in cat_results if metric in r]
            if vals:
                summary[f"{cat_name}_{metric}_mean"] = float(np.mean(vals))
                summary[f"{cat_name}_{metric}_std"] = float(np.std(vals))

    # Overall: SVD vs FFT parameter efficiency comparison
    svd_params_total = sum(r.get("svd_params_90", 0) for r in active)
    fft_params_total = sum(r.get("fft_params_90", 0) for r in active)
    dct_params_total = sum(r.get("dct_params_90", 0) for r in active if "dct_params_90" in r)

    summary["total_svd_params_90"] = int(svd_params_total)
    summary["total_fft_params_90"] = int(fft_params_total)
    summary["total_dct_params_90"] = int(dct_params_total)

    # Efficiency ratio: how many fewer params does SVD need vs FFT for same quality?
    if fft_params_total > 0:
        summary["svd_vs_fft_ratio"] = svd_params_total / fft_params_total
    if dct_params_total > 0:
        summary["svd_vs_dct_ratio"] = svd_params_total / dct_params_total

    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Phase 3 Diagnostic: Weight Update Structure Analysis")
    logger.info("=" * 80)

    # Step 1: Full fine-tuning to get ground-truth ΔW
    pretrained, finetuned, train_metrics = train_full_finetune(
        model_name="bert-base-uncased",
        task_name="mrpc",
        num_epochs=3,
        lr=2e-5,
        batch_size=32,
    )

    logger.info(f"Training complete. Best F1: {train_metrics['best_f1']:.4f}")

    # Step 2: Analyze ΔW structure
    logger.info("Analyzing weight update structure...")
    per_layer_results = analyze_delta_w(pretrained, finetuned)

    # Step 3: Summarize
    summary = summarize_results(per_layer_results)

    # Save results
    output = {
        "train_metrics": train_metrics,
        "per_layer": per_layer_results,
        "summary": summary,
    }

    output_path = os.path.join(RESULTS_DIR, "weight_update_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print key findings
    logger.info("=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)
    logger.info(f"Total weight matrices analyzed: {summary.get('total_matrices', 0)}")
    logger.info(f"Total params needed (SVD, 90% energy): {summary.get('total_svd_params_90', 'N/A')}")
    logger.info(f"Total params needed (FFT, 90% energy): {summary.get('total_fft_params_90', 'N/A')}")
    logger.info(f"Total params needed (DCT, 90% energy): {summary.get('total_dct_params_90', 'N/A')}")
    logger.info(f"SVD/FFT efficiency ratio: {summary.get('svd_vs_fft_ratio', 'N/A')}")

    for cat in ["attention_qkv", "attention_output", "ffn_intermediate", "ffn_output"]:
        rank_90 = summary.get(f"{cat}_svd_rank_90_mean", "N/A")
        rank_90_frac = summary.get(f"{cat}_svd_rank_90_frac_mean", "N/A")
        fft_frac = summary.get(f"{cat}_fft_k_90_frac_mean", "N/A")
        top1 = summary.get(f"{cat}_svd_top1_energy_frac_mean", "N/A")
        logger.info(f"\n{cat}:")
        logger.info(f"  SVD rank for 90% energy: {rank_90} (fraction: {rank_90_frac})")
        logger.info(f"  FFT coeffs for 90% energy (fraction): {fft_frac}")
        logger.info(f"  Top-1 singular value energy: {top1}")

    # Cleanup
    del pretrained, finetuned
    gc.collect()


if __name__ == "__main__":
    main()
