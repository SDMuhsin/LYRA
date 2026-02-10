"""Diagnostic: compare gradient norms for Spectral vs VeRA on CoLA."""
import sys, math, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import default_data_collator
import datasets, numpy as np

device = torch.device("cuda:0")

def setup_model(method):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = model.float()
    if method == "spectral":
        sys.path.insert(0, "src")
        from spectral_adapter import get_spectral_adapter_model
        model = get_spectral_adapter_model(model, ["query","key","value","dense"], p=16, q=16, scaling=1.0)
    elif method == "vera":
        from peft import VeraConfig, get_peft_model, TaskType
        cfg = VeraConfig(r=128, target_modules=["query","key","value","dense"],
                         vera_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS, d_initial=0.1)
        model = get_peft_model(model, cfg)
    return model.to(device)

def get_cola_batch():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = datasets.load_dataset("glue", "cola", split="train[:64]")
    def preprocess(ex):
        result = tokenizer(ex["sentence"], padding="max_length", max_length=128, truncation=True)
        result["labels"] = ex["label"]
        return result
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    loader = DataLoader(ds, batch_size=32, collate_fn=default_data_collator)
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}

batch = get_cola_batch()

for method in ["spectral", "vera"]:
    model = setup_model(method)
    model.train()

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Collect gradient stats
    all_grads = []
    adapter_grads = []
    classifier_grads = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach()
            gnorm = g.norm().item()
            all_grads.append((name, gnorm, param.numel(), g.abs().max().item(), g.abs().mean().item()))
            if 'classifier' in name:
                classifier_grads.append(gnorm)
            else:
                adapter_grads.append(gnorm)

    total_norm = math.sqrt(sum(gnorm**2 for _, gnorm, _, _, _ in all_grads))
    adapter_norm = math.sqrt(sum(g**2 for g in adapter_grads))
    classifier_norm = math.sqrt(sum(g**2 for g in classifier_grads))

    print(f"\n{'='*70}")
    print(f"Method: {method.upper()}")
    print(f"{'='*70}")
    print(f"Total gradient norm:      {total_norm:.4f}")
    print(f"Adapter gradient norm:    {adapter_norm:.4f}")
    print(f"Classifier gradient norm: {classifier_norm:.4f}")
    print(f"Clip threshold: 1.0  →  clipping ratio: {total_norm/1.0:.2f}x")
    print(f"\nTop 10 parameters by gradient norm:")
    all_grads.sort(key=lambda x: x[1], reverse=True)
    for name, gnorm, numel, gmax, gmean in all_grads[:10]:
        short = name[-60:] if len(name) > 60 else name
        print(f"  {gnorm:8.4f}  (max={gmax:.4f} mean={gmean:.6f} n={numel:>6})  {short}")

    print(f"\nBottom 5 adapter params by gradient norm:")
    adapter_only = [(n,g,nu,mx,mn) for n,g,nu,mx,mn in all_grads if 'classifier' not in n]
    for name, gnorm, numel, gmax, gmean in adapter_only[-5:]:
        short = name[-60:] if len(name) > 60 else name
        print(f"  {gnorm:8.4f}  (max={gmax:.4f} mean={gmean:.6f} n={numel:>6})  {short}")

    del model
    torch.cuda.empty_cache()
