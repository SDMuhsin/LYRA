"""Diagnostic: measure per-step output change magnitude for Spectral vs VeRA.

Root cause hypothesis: Spectral's norm-preserving DCT causes larger model
output changes per optimizer step compared to VeRA (dampened by d_initial=0.1),
leading to training instability on sensitive tasks like CoLA.

This script measures:
1. Logit change (L2 norm) after each optimizer step
2. Prediction flip rate (how many samples change predicted class)
3. Coefficient/weight change magnitude per step
"""
import sys, math, torch, torch.nn as nn
import numpy as np
sys.path.insert(0, "src")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
import datasets

device = torch.device("cuda:0")

def setup_model(method):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = model.float()
    if method == "spectral":
        from spectral_adapter import get_spectral_adapter_model
        model = get_spectral_adapter_model(model, ["query","key","value","dense"], p=16, q=16, scaling=1.0)
    elif method == "spectral_s0.1":
        from spectral_adapter import get_spectral_adapter_model
        model = get_spectral_adapter_model(model, ["query","key","value","dense"], p=16, q=16, scaling=0.1)
    elif method == "vera":
        from peft import VeraConfig, get_peft_model, TaskType
        cfg = VeraConfig(r=128, target_modules=["query","key","value","dense"],
                         vera_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS, d_initial=0.1)
        model = get_peft_model(model, cfg)
    return model.to(device)

def get_cola_loader():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = datasets.load_dataset("glue", "cola", split="train[:256]")
    def preprocess(ex):
        result = tokenizer(ex["sentence"], padding="max_length", max_length=128, truncation=True)
        result["labels"] = ex["label"]
        return result
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    return DataLoader(ds, batch_size=32, collate_fn=default_data_collator)

loader = get_cola_loader()
eval_batch = next(iter(loader))
eval_batch = {k: v.to(device) for k, v in eval_batch.items()}

NUM_STEPS = 20

for method in ["spectral", "spectral_s0.1", "vera"]:
    print(f"\n{'='*70}")
    print(f"Method: {method}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    model = setup_model(method)

    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=0.002, weight_decay=0.01)
    # Simple linear schedule matching the real training
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=7980)

    model.train()
    batches = list(loader)

    logit_changes = []
    flip_rates = []
    adapter_param_changes = []
    grad_norms_total = []
    grad_norms_adapter = []

    # Get initial logits
    model.eval()
    with torch.no_grad():
        prev_logits = model(**eval_batch).logits.detach().clone()
    model.train()

    # Snapshot adapter parameters
    prev_adapter_params = {}
    for name, p in model.named_parameters():
        if p.requires_grad and 'classifier' not in name:
            prev_adapter_params[name] = p.data.clone()

    for step_i in range(NUM_STEPS):
        batch = batches[step_i % len(batches)]
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Record gradient norms before clipping
        total_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        adapter_gnorm = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and 'classifier' not in name:
                adapter_gnorm += p.grad.norm().item()**2
        adapter_gnorm = math.sqrt(adapter_gnorm)
        grad_norms_total.append(total_gnorm)
        grad_norms_adapter.append(adapter_gnorm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Measure logit change
        model.eval()
        with torch.no_grad():
            new_logits = model(**eval_batch).logits.detach()
        model.train()

        logit_diff = (new_logits - prev_logits).norm().item()
        prev_preds = prev_logits.argmax(dim=-1)
        new_preds = new_logits.argmax(dim=-1)
        flip_rate = (prev_preds != new_preds).float().mean().item()

        logit_changes.append(logit_diff)
        flip_rates.append(flip_rate)

        # Measure adapter parameter change
        total_param_change = 0.0
        for name, p in model.named_parameters():
            if name in prev_adapter_params:
                total_param_change += (p.data - prev_adapter_params[name]).norm().item()**2
                prev_adapter_params[name] = p.data.clone()
        adapter_param_changes.append(math.sqrt(total_param_change))

        prev_logits = new_logits.clone()

    # Print results
    print(f"\nPer-step metrics (first {NUM_STEPS} steps):")
    print(f"{'Step':>4}  {'Logit Δ':>10}  {'Flip %':>8}  {'Adapter Δθ':>12}  {'Grad norm':>10}  {'Adapter gnorm':>13}")
    for i in range(NUM_STEPS):
        print(f"{i:4d}  {logit_changes[i]:10.4f}  {flip_rates[i]*100:7.1f}%  {adapter_param_changes[i]:12.6f}  {grad_norms_total[i]:10.4f}  {grad_norms_adapter[i]:13.6f}")

    print(f"\nSummary:")
    print(f"  Mean logit change per step:     {np.mean(logit_changes):.4f}")
    print(f"  Mean flip rate per step:         {np.mean(flip_rates)*100:.1f}%")
    print(f"  Mean adapter param change/step:  {np.mean(adapter_param_changes):.6f}")
    print(f"  Mean total gradient norm (pre-clip): {np.mean(grad_norms_total):.4f}")
    print(f"  Mean adapter gradient norm:      {np.mean(grad_norms_adapter):.6f}")
    print(f"  Output sensitivity (logit_Δ / adapter_Δθ): {np.mean(logit_changes) / (np.mean(adapter_param_changes) + 1e-12):.2f}")

    del model, optimizer
    torch.cuda.empty_cache()
