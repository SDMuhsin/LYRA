# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning 🤗 Transformers models for sequence-classification on GLUE, running the
same training five times with seeds 41-45 (“median-of-five”, Mo5).
After the five runs finish we log **only the median** task-performance numbers
to `./results/mo5_glue.csv`; ancillary metrics (memory, timing, …) come from the
**first** seed’s run. The “seed” column in the CSV is literally the string
`"41,42,43,44,45"`.
"""
import argparse
import builtins
import copy
import csv
import gc
import json
import logging
import math
import operator
import os
import random
import statistics
import time
from functools import reduce
from pathlib import Path
from typing import Dict, List

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Import GaLore optimizers (standard ones)
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import bitsandbytes as bnb

# Try to import GALE optimizers (optional, from custom fork)
try:
    from galore_torch import GALE_AdamW, GALE_Adafactor, GALE_AdamW8bit, SwiftGaLoreAdamW, GALE_Lion
    GALE_AVAILABLE = True
except ImportError:
    GALE_AVAILABLE = False
    # Provide dummy classes for when GALE is not available
    class _DummyOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("GALE optimizers not available. Install galore-torch from custom fork.")
    GALE_AdamW = GALE_Adafactor = GALE_AdamW8bit = SwiftGaLoreAdamW = GALE_Lion = _DummyOptimizer

# Import Lion optimizer
from lion_pytorch import Lion

# Import AdapterHub
import adapters
from adapters import LoRAConfig, IA3Config, PrefixTuningConfig
from filelock import FileLock

# Import PEFT library for DoRA, VeRA, FourierFT, and AdaLoRA
from peft import (
    LoraConfig as PeftLoraConfig,
    VeraConfig,
    FourierFTConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType
)

# Import GB-VeRA (our gradient-balanced VeRA implementation)
from gbvera import get_gbvera_model, GBVeraModel

# Import Spectral Adapter (Truncated DCT Factored Adaptation)
from spectral_adapter import get_spectral_adapter_model, SpectralAdapterModel

# Import DyLoRA (Dynamic Low-Rank Adaptation)
from dylora import get_dylora_model, DyLoRAModel



torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

###############################################################################
#                                   constants                                 #
###############################################################################
SEEDS: List[int] = [41, 42, 43, 44, 45]
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "mo53_glue.csv")
LOCK_FILE_PATH = os.path.join(RESULTS_DIR, "mo53_glue.csv.lock") 
_METRIC_FOR_TASK = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
    "cb": "f1",
}

###############################################################################
#                                   helpers                                   #
###############################################################################
logger = logging.getLogger(__name__)


def _primary_metric(task_name: str, metric_dict: dict) -> float:
    key = _METRIC_FOR_TASK.get(task_name, "accuracy")
    return metric_dict.get(key, float("-inf"))


def _load_results_df(columns: List[str]) -> pd.DataFrame:
    if os.path.isfile(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns]
    return pd.DataFrame(columns=columns)


def _upsert_result(df: pd.DataFrame, comb_cols: List[str], row_dict: Dict) -> pd.DataFrame:
    mask = reduce(
        operator.and_, [(df[col] == row_dict[col]) for col in comb_cols], pd.Series(True, index=df.index)
    )
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    return df


###############################################################################
#                                   data-keys                                 #
###############################################################################
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "anli_r1": ("premise", "hypothesis"),
}

###############################################################################
#                                  arg-parsing                                #
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a GLUE task (Mo5 variant)")

    # Model and Data Arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--load_pretrained_model", type=str, default=None, help="Path to a checkpoint to load model weights from.")
    parser.add_argument("--task_name", type=str, required=True, choices=list(task_to_keys.keys()))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")

    # Training Hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use (e.g., 'adamw', 'galore_adamw', 'adamw-lora').")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per-device batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=None, help="Effective total batch size. Overrides gradient_accumulation_steps if set.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as a ratio of the max learning rate.")
    parser.add_argument("--grad_clipping", type=float, default=1.0, help="Gradient clipping value. 0.0 to disable.")
    parser.add_argument("--beta1", type=float, default=0.0, help="Beta1 for Adam-like optimizers (e.g., Adafactor).")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32", help="Data type for model training (bfloat16, float16, float32).")

    # GaLore / GALE Specific Arguments
    parser.add_argument("--rank", type=int, default=128, help="Rank for GaLore/GALE projection matrices.")
    parser.add_argument("--update_proj_gap", type=int, default=50, help="Frequency of updating GaLore/GALE projection matrices.")
    parser.add_argument("--galore_scale", type=float, default=1.0, help="Scaling factor for GaLore.")
    parser.add_argument("--proj_type", type=str, default="std", help="Projection type for GaLore.")

    # AdapterHub Specific Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--prefix_bottleneck_size", type=int, default=256, help="Prefix Tuning bottleneck size.")
    parser.add_argument("--lora_all_modules", action="store_true", help="Apply LoRA to all supported linear layers.")

    # PEFT-specific Arguments (DoRA, VeRA, FourierFT)
    parser.add_argument("--dora_r", type=int, default=16, help="DoRA rank (typically half of LoRA rank).")
    parser.add_argument("--dora_alpha", type=int, default=32, help="DoRA alpha scaling parameter.")
    parser.add_argument("--dora_dropout", type=float, default=0.05, help="DoRA dropout probability.")
    parser.add_argument("--vera_r", type=int, default=256, help="VeRA rank (typically higher than LoRA).")
    parser.add_argument("--vera_dropout", type=float, default=0.0, help="VeRA dropout probability.")
    parser.add_argument("--vera_d_initial", type=float, default=0.1, help="VeRA initial value for scaling vectors.")
    parser.add_argument("--vera_projection_prng_key", type=int, default=0, help="VeRA random seed for projection initialization.")
    parser.add_argument("--fourierft_n_frequency", type=int, default=1000, help="FourierFT number of learnable frequency components.")
    parser.add_argument("--fourierft_scaling", type=float, default=150.0, help="FourierFT scaling parameter (100-150 for GLUE/NLU, 300 for LLaMA/ViT).")
    parser.add_argument("--fourierft_random_loc_seed", type=int, default=777, help="FourierFT random seed for frequency selection.")

    # GB-VeRA Specific Arguments
    parser.add_argument("--gbvera_r", type=int, default=256, help="GB-VeRA rank (typically same as VeRA, default 256).")
    parser.add_argument("--gbvera_d_initial", type=float, default=0.1, help="GB-VeRA initial value for λ_d (default 0.1).")
    parser.add_argument("--gbvera_b_initial", type=float, default=0.01, help="GB-VeRA initial value for λ_b (default 0.01, non-zero to fix bootstrap).")
    parser.add_argument("--gbvera_dropout", type=float, default=0.0, help="GB-VeRA dropout probability.")
    parser.add_argument("--gbvera_projection_prng_key", type=int, default=0, help="GB-VeRA random seed for projection initialization.")

    # AdaLoRA Specific Arguments
    parser.add_argument("--adalora_init_r", type=int, default=12, help="AdaLoRA initial rank (before pruning).")
    parser.add_argument("--adalora_target_r", type=int, default=4, help="AdaLoRA target rank (after pruning).")
    parser.add_argument("--adalora_alpha", type=int, default=8, help="AdaLoRA alpha scaling parameter.")
    parser.add_argument("--adalora_dropout", type=float, default=0.0, help="AdaLoRA dropout probability.")
    parser.add_argument("--adalora_tinit", type=int, default=200, help="AdaLoRA: initial warmup steps (no pruning). Paper default=200.")
    parser.add_argument("--adalora_tfinal", type=int, default=200, help="AdaLoRA: final steps (no pruning). Paper default=200.")
    parser.add_argument("--adalora_deltaT", type=int, default=10, help="AdaLoRA: interval between rank allocation steps. Paper default=10.")
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=0.5, help="AdaLoRA: orthogonality regularization weight.")

    # DyLoRA Specific Arguments
    parser.add_argument("--dylora_r", type=int, default=8, help="DyLoRA max rank (trains across ranks 1..r).")
    parser.add_argument("--dylora_alpha", type=int, default=16, help="DyLoRA alpha scaling parameter.")
    parser.add_argument("--dylora_dropout", type=float, default=0.0, help="DyLoRA dropout probability.")

    # Spectral Adapter (Truncated DCT Factored Adaptation) Arguments
    parser.add_argument("--spectral_p", type=int, default=32, help="Spectral adapter: number of DCT basis vectors for output dimension.")
    parser.add_argument("--spectral_q", type=int, default=32, help="Spectral adapter: number of DCT basis vectors for input dimension.")
    parser.add_argument("--spectral_scaling", type=float, default=1.0, help="Spectral adapter: scaling factor for adapter output.")
    parser.add_argument("--spectral_dropout", type=float, default=0.0, help="Spectral adapter: dropout probability.")
    parser.add_argument("--spectral_d_initial", type=float, default=0.0, help="Spectral adapter: if > 0, initialize coefficients with N(0, d_initial) instead of zeros.")
    parser.add_argument("--spectral_target_modules", type=str, default=None, help="Spectral adapter: comma-separated list of target module names (e.g., 'query,value'). If None, uses architecture defaults.")
    parser.add_argument("--spectral_freq_mode", type=str, default="contiguous", choices=["contiguous", "geometric", "geometric_half", "hybrid"], help="Spectral adapter: frequency selection strategy. 'contiguous' uses [0..k-1], 'geometric' uses power-spaced indices over [0, d//2], 'hybrid' uses 3k/4 contiguous low + k/4 geometric high.")
    parser.add_argument("--spectral_freq_exponent", type=float, default=2.0, help="Spectral adapter: exponent for geometric spacing (default 2.0=quadratic). 1.0=linear/uniform, 3.0=cubic/denser low-freq.")
    parser.add_argument("--spectral_factored_rank", type=int, default=0, help="Spectral adapter: if > 0, factor S = A(p,r)@B(r,q) for wider freq coverage. Params per module = p*r + r*q. 0 = dense S (default).")
    parser.add_argument("--spectral_learn_scaling", action="store_true", default=False, help="Spectral adapter: if set, each module gets a learnable log-space scaling parameter (+1 param/module).")

    # Generic target-module override (applies to all adapter methods)
    parser.add_argument("--adapter_target_modules", type=str, default=None,
        help="Comma-separated target module names, overrides architecture defaults")

    # Execution & Benchmarking Arguments
    parser.add_argument("--name", type=str, default="glue_finetuning_run", help="A name for this training run.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="(ignored, script uses fixed seeds 41-45)")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--per_layer_opt", action="store_true", help="Enable per-layer optimization (no retaining grad mode) where gradients are applied immediately layer by layer.")

    # Hub / Checkpointing Arguments
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory at the cost of slower backward pass.")

    args = parser.parse_args()

    if args.total_batch_size:
        assert args.total_batch_size % args.per_device_train_batch_size == 0, "total_batch_size must be divisible by per_device_train_batch_size"
        args.gradient_accumulation_steps = args.total_batch_size // args.per_device_train_batch_size
    # Note: final total_batch_size is calculated in run_single_seed

    # Handle AdapterHub/PEFT/Optimizer method detection
    args.adapter_method = None
    args.optimizer_base = args.optimizer.lower()
    # AdapterHub methods: lora, ia3, prefix
    # PEFT methods: dora, vera, fourierft
    # Custom methods: gbvera (gradient-balanced VeRA)
    adapter_methods = ['lora', 'ia3', 'prefix', 'dora', 'vera', 'fourierft', 'gbvera', 'spectral', 'adalora', 'dylora']
    for method in adapter_methods:
        suffix = f'-{method}'
        if args.optimizer.lower().endswith(suffix):
            args.adapter_method = method
            args.optimizer_base = args.optimizer.lower().replace(suffix, '')
            break

    return args

###############################################################################
#                               memory accounting                             #
###############################################################################
def mib(x: int) -> float:
    """Converts bytes to MiB."""
    return x / 1024 ** 2

def calculate_theoretical_memory(model: nn.Module, args: argparse.Namespace) -> float:
    """
    Calculates the theoretical memory usage in MiB for various optimizers including Adam(W), Adafactor,
    AdamW8bit, LoRA, GaLore, GALE, Lion, IA³, and Prefix-Tuning.
    Assumes bf16 (2 bytes per parameter) for model weights and optimizer states.
    Returns 0.0 for unsupported configurations as a placeholder.
    """
    # Supported optimizers and their variants
    is_galore_or_gale = 'galore' in args.optimizer_base or 'gale' in args.optimizer_base
    is_adam = args.optimizer_base in ['adam', 'adamw']
    is_adafactor = args.optimizer_base in ['adafactor']
    is_adamw8bit = args.optimizer_base in ['adam8bit', 'adamw8bit']
    is_lion = args.optimizer_base in ['lion']

    if not (is_galore_or_gale or is_adam or is_adafactor or is_adamw8bit or is_lion):
        return 0.0

    total_model_params = sum(p.numel() for p in model.parameters())
    optimizer_state_params = 0
    # For Adam/AdamW, optimizer states are 2x the number of trainable parameters (momentum + variance)
    # For Lion, optimizer states are 1x the number of trainable parameters (only momentum, exp_avg)
    optimizer_state_multiplier = 1 if is_lion else 2

    if is_galore_or_gale:
        # Check if this is GALE or GaLore
        is_gale = 'gale' in args.optimizer.lower()
        
        if is_gale:
            # GALE: Memory = Full model params + GALE optimizer states (stored in low-rank space)
            gale_param_ids = set()
            # For BERT/RoBERTa models, target attention and feedforward layers
            target_modules = ["attention", "intermediate", "output"] if "llama" not in args.model_name_or_path.lower() else ["attn", "mlp"]

            # Calculate GALE-specific optimizer state size
            # GALE stores optimizer states (exp_avg, exp_avg_sq) in low-rank space
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                    m, n = module.weight.shape
                    # GALE projects gradient to low-rank space and stores optimizer states there
                    # For matrix m×n, the low-rank gradient has dimensions:
                    # - If m >= n: gradient is m×r, so optimizer states are 2×(m×r) = 2mr
                    # - If m < n: gradient is r×n, so optimizer states are 2×(r×n) = 2rn
                    if m >= n:
                        low_rank_size = m * args.rank
                    else:
                        low_rank_size = args.rank * n
                    optimizer_state_params += optimizer_state_multiplier * low_rank_size
                    gale_param_ids.add(id(module.weight))
            
            # Add standard optimizer states for other trainable parameters (e.g., embeddings, LayerNorms)
            for p in model.parameters():
                if p.requires_grad and id(p) not in gale_param_ids:
                    optimizer_state_params += optimizer_state_multiplier * p.numel()
        else:
            # GaLore: Memory = Full model params + GaLore optimizer states (stored in low-rank space)
            # GaLore actually stores optimizer states in low-rank space, same as GALE
            galore_param_ids = set()
            # For BERT/RoBERTa models, target attention and feedforward layers
            target_modules = ["attention", "intermediate", "output"] if "llama" not in args.model_name_or_path.lower() else ["attn", "mlp"]

            # Calculate GaLore-specific optimizer state size
            # GaLore stores optimizer states (exp_avg, exp_avg_sq) in low-rank space
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                    m, n = module.weight.shape
                    # GaLore projects gradient to low-rank space and stores optimizer states there
                    # For matrix m×n, the low-rank gradient has dimensions:
                    # - If m >= n: gradient is m×r, so optimizer states are 2×(m×r) = 2mr
                    # - If m < n: gradient is r×n, so optimizer states are 2×(r×n) = 2rn
                    if m >= n:
                        low_rank_size = m * args.rank
                    else:
                        low_rank_size = args.rank * n
                    optimizer_state_params += optimizer_state_multiplier * low_rank_size
                    galore_param_ids.add(id(module.weight))
            
            # Add standard optimizer states for other trainable parameters (e.g., embeddings, LayerNorms)
            for p in model.parameters():
                if p.requires_grad and id(p) not in galore_param_ids:
                    optimizer_state_params += optimizer_state_multiplier * p.numel()

    elif is_adam or is_adafactor or is_adamw8bit or is_lion:
        # Handle different adapter methods for all supported optimizers
        if args.adapter_method == 'lora':
            # LoRA: Only LoRA parameters are trainable, very small memory footprint
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                # Adafactor can use factored second moments for 2D parameters
                # For simplicity, we assume non-factored mode (similar to Adam)
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                # AdamW8bit uses 8-bit quantized states, but we calculate in full precision equivalent
                # The actual memory usage is lower, but we use full precision for theoretical calculation
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'dora':
            # DoRA: Similar to LoRA but with magnitude decomposition
            # Memory is similar to LoRA with slightly more parameters for magnitude vectors
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'vera':
            # VeRA: Very few trainable parameters (only scaling vectors)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'gbvera':
            # GB-VeRA: Same parameter count as VeRA (μ_d and μ_b instead of λ_d and λ_b)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'fourierft':
            # FourierFT: Extremely few trainable parameters (spectral coefficients)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'adalora':
            # AdaLoRA: SVD-parameterized LoRA with adaptive rank allocation
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'dylora':
            # DyLoRA: Dynamic LoRA (same param count as LoRA at max rank)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'ia3':
            # IA³: Only scaling vectors are trainable, optimizer states for scaling vectors only
            ia3_optimizer_params = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    m, n = module.weight.shape
                    # IA³ introduces scaling vector of dimension n
                    # Optimizer states: 2 * n (momentum + variance for scaling vector)
                    ia3_optimizer_params += optimizer_state_multiplier * n
            optimizer_state_params = ia3_optimizer_params
            
        elif args.adapter_method == 'prefix':
            # Prefix-Tuning: Prefix parameters are trainable
            prefix_optimizer_params = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    m, n = module.weight.shape
                    p = args.prefix_bottleneck_size
                    # Prefix parameters: 2pn (for key and value prefixes)
                    # Optimizer states: 2 * (2pn) = 4pn (momentum + variance for prefix parameters)
                    prefix_optimizer_params += optimizer_state_multiplier * (2 * p * n)
            optimizer_state_params = prefix_optimizer_params
            
        else:
            # Full Fine-Tuning: All parameters are trainable
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                # Adafactor memory calculation
                # For 2D parameters (matrices), Adafactor can use factored second moments
                # This reduces memory from O(mn) to O(m+n) for an m×n matrix
                # For 1D parameters, it uses standard second moments
                adafactor_state_params = 0
                for p in model.parameters():
                    if p.requires_grad:
                        if len(p.shape) >= 2:  # 2D or higher dimensional parameters
                            # Factored mode: row and column statistics
                            # exp_avg_sq_row: shape[:-1] elements
                            # exp_avg_sq_col: shape[:-2] + shape[-1:] elements
                            row_size = 1
                            for dim in p.shape[:-1]:
                                row_size *= dim
                            col_size = 1
                            for dim in p.shape[:-2]:
                                col_size *= dim
                            col_size *= p.shape[-1]
                            factored_size = row_size + col_size
                            
                            # Add first moment if beta1 is used
                            if hasattr(args, 'beta1') and args.beta1 and args.beta1 > 0:
                                adafactor_state_params += p.numel()  # exp_avg
                            adafactor_state_params += factored_size  # factored second moments
                        else:
                            # Non-factored mode for 1D parameters
                            if hasattr(args, 'beta1') and args.beta1 and args.beta1 > 0:
                                adafactor_state_params += p.numel()  # exp_avg
                            adafactor_state_params += p.numel()  # exp_avg_sq
                optimizer_state_params = adafactor_state_params
            elif is_adamw8bit:
                # AdamW8bit uses quantized states, but we calculate theoretical full precision
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params
    
    # Total parameters for memory calculation = model weights + optimizer states
    total_theoretical_params = total_model_params + optimizer_state_params
    
    # Convert to MiB assuming 2 bytes per parameter (BF16)
    bytes_per_mib = 1024**2
    memory_mib = (total_theoretical_params * 2) / bytes_per_mib
    
    return memory_mib

@torch.no_grad()
def get_memory_breakdown(model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           device: torch.device) -> dict:
    """
    Returns a breakdown of memory usage in MiB.
    """
    stats = {}
    if device.type == "cuda":
        # Model Parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        stats['param_mem_mib'] = mib(param_bytes)

        # Optimizer State
        opt_bytes = 0
        if optimizer and hasattr(optimizer, 'state') and optimizer.state:
            for state in optimizer.state.values():
                for t in state.values():
                    if torch.is_tensor(t):
                        opt_bytes += t.numel() * t.element_size()
        stats['opt_mem_mib'] = mib(opt_bytes)

        # CUDA Memory Stats
        stats['peak_memory_mib'] = mib(torch.cuda.max_memory_allocated(device))
        stats['allocated_memory_mib'] = mib(torch.cuda.memory_allocated(device))
    return stats

###############################################################################
#                             single-seed training loop                         #
###############################################################################
def run_single_seed(base_args: argparse.Namespace, seed: int):
    """
    Execute **one** full training run with the given `seed`.
    """
    args = copy.deepcopy(base_args)
    args.seed = seed
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, f"seed_{seed}")

    # --- Device and Seed Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Calculate total batch size
    if not args.total_batch_size:
        # Assuming a single device (num_processes = 1)
        args.total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"[seed {seed}] Running on device: {device}")
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.push_to_hub:
        repo_name = args.hub_model_id or Path(args.output_dir).absolute().name
        repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
        repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Data Loading ---
    if args.task_name in ("boolq", "cb"):
        raw_datasets = load_dataset("super_glue", args.task_name)
    elif args.task_name == "anli_r1":
        _anli = load_dataset("facebook/anli")
        # Remap ANLI R1 splits to standard names
        from datasets import DatasetDict
        raw_datasets = DatasetDict({
            "train": _anli["train_r1"],
            "validation": _anli["dev_r1"],
            "test": _anli["test_r1"],
        })
    else:
        raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # --- Model Initialization ---
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )
    
    if (args.download_only):
       logger.info("DOWNLOAD ONLY (passed via --download_only flag") 
       exit()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- Dtype, Adapter, and Device Setup ---
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32
    
    # Cast model to the correct dtype before adapter init or moving to device
    if dtype != torch.float32:
        model.to(dtype=dtype)
    
    if args.adapter_method:
        # PEFT methods: dora, vera, fourierft
        # Custom methods: gbvera, spectral
        peft_methods = ['dora', 'vera', 'fourierft', 'adalora']
        custom_methods = ['gbvera', 'spectral', 'dylora']

        if args.adapter_method == 'spectral':
            # Use our Truncated DCT Factored Adaptation
            logger.info(f"Initializing model for Spectral Adapter (Truncated DCT) training...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.spectral_target_modules:
                target_modules = [m.strip() for m in args.spectral_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Spectral adapter target modules: {target_modules}")

            model = get_spectral_adapter_model(
                model=model,
                target_modules=target_modules,
                p=args.spectral_p,
                q=args.spectral_q,
                scaling=args.spectral_scaling,
                dropout=args.spectral_dropout,
                d_initial=args.spectral_d_initial,
                freq_mode=args.spectral_freq_mode,
                freq_exponent=args.spectral_freq_exponent,
                factored_rank=args.spectral_factored_rank,
                learn_scaling=args.spectral_learn_scaling,
            )

            logger.info(f"Successfully applied Spectral Adapter to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'dylora':
            # Use our custom DyLoRA implementation
            logger.info(f"Initializing model for DyLoRA training (custom implementation)...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            model = get_dylora_model(
                model=model,
                target_modules=target_modules,
                r=args.dylora_r,
                alpha=args.dylora_alpha,
                dropout=args.dylora_dropout,
            )

            logger.info(f"Successfully applied DyLoRA to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'gbvera':
            # Use our custom GB-VeRA implementation
            logger.info(f"Initializing model for GB-VeRA training (custom implementation)...")

            # Determine target modules based on model architecture
            if "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "v_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]

            model = get_gbvera_model(
                model=model,
                target_modules=target_modules,
                r=args.gbvera_r,
                d_initial=args.gbvera_d_initial,
                b_initial=args.gbvera_b_initial,
                dropout=args.gbvera_dropout,
                projection_prng_key=args.gbvera_projection_prng_key,
            )

            logger.info(f"Successfully applied GB-VeRA to model.")
            model.print_trainable_parameters()

        elif args.adapter_method in peft_methods:
            logger.info(f"Initializing model for {args.adapter_method.upper()} training with PEFT library...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                # For BERT/RoBERTa models used in GLUE
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                # For GPT-2 models
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                # For LLaMA models
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                # For OPT models (separate Q/K/V/out_proj + FFN)
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                # Default: try common attention projection names
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            peft_config = None
            if args.adapter_method == 'dora':
                peft_config = PeftLoraConfig(
                    r=args.dora_r,
                    lora_alpha=args.dora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.dora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    use_dora=True  # Enable DoRA
                )
            elif args.adapter_method == 'vera':
                peft_config = VeraConfig(
                    r=args.vera_r,
                    target_modules=target_modules,
                    vera_dropout=args.vera_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    save_projection=True,
                    projection_prng_key=args.vera_projection_prng_key,
                    d_initial=args.vera_d_initial
                )
            elif args.adapter_method == 'fourierft':
                peft_config = FourierFTConfig(
                    n_frequency=args.fourierft_n_frequency,
                    target_modules=target_modules,
                    task_type=TaskType.SEQ_CLS,
                    scaling=args.fourierft_scaling,
                    random_loc_seed=args.fourierft_random_loc_seed
                )
            elif args.adapter_method == 'adalora':
                # Pre-compute total training steps for AdaLoRA's rank allocation schedule
                n_train = len(raw_datasets["train"])
                est_steps_per_epoch = math.ceil(n_train / args.per_device_train_batch_size / args.gradient_accumulation_steps)
                est_total_steps = args.max_train_steps if args.max_train_steps else est_steps_per_epoch * args.num_train_epochs
                logger.info(f"AdaLoRA: estimated total_step={est_total_steps} for rank allocation schedule")

                peft_config = AdaLoraConfig(
                    init_r=args.adalora_init_r,
                    target_r=args.adalora_target_r,
                    lora_alpha=args.adalora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.adalora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    total_step=est_total_steps,
                    tinit=args.adalora_tinit,
                    tfinal=args.adalora_tfinal,
                    deltaT=args.adalora_deltaT,
                    orth_reg_weight=args.adalora_orth_reg_weight,
                )

            if peft_config:
                model = get_peft_model(model, peft_config)
                logger.info(f"Successfully applied {args.adapter_method.upper()} to model via PEFT.")
                model.print_trainable_parameters()
        else:
            # AdapterHub methods: lora, ia3, prefix
            logger.info(f"Initializing model for {args.adapter_method.upper()} training with AdapterHub...")
            adapters.init(model)

            adapter_config = None
            if args.adapter_method == 'lora':
                if args.adapter_target_modules:
                    modules = [m.strip() for m in args.adapter_target_modules.split(",")]
                    attn_matrices = []
                    for m in modules:
                        if m in ("query", "q_proj"):
                            attn_matrices.append("q")
                        elif m in ("key", "k_proj"):
                            attn_matrices.append("k")
                        elif m in ("value", "v_proj"):
                            attn_matrices.append("v")
                    intermediate_lora = any(m in ("dense", "intermediate", "fc1") for m in modules)
                    output_lora = any(m in ("dense", "output", "fc2") for m in modules)
                    adapter_config = LoRAConfig(
                        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                        attn_matrices=attn_matrices or ["q", "v"],
                        intermediate_lora=intermediate_lora, output_lora=output_lora,
                    )
                else:
                    adapter_config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            elif args.adapter_method == 'ia3':
                adapter_config = IA3Config()
            elif args.adapter_method == 'prefix':
                adapter_config = PrefixTuningConfig(bottleneck_size=args.prefix_bottleneck_size)

            if adapter_config:
                adapter_name = f"{args.adapter_method}_adapter"
                model.add_adapter(adapter_name, config=adapter_config)
                model.train_adapter(adapter_name)
                model.set_active_adapters(adapter_name)

                # Cast model again after adding adapters to ensure new params are also in the correct dtype
                if dtype != torch.float32:
                    model.to(dtype=dtype)
                logger.info(f"Successfully added and enabled {args.adapter_method.upper()} adapter for training.")

    model.to(device)

    # --- Enable Gradient Checkpointing ---
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info(f"[seed {seed}] Gradient checkpointing enabled")
        else:
            logger.warning(f"[seed {seed}] Model does not support gradient checkpointing, skipping")

    # --- Dataset Preprocessing ---
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, desc="Tokenising",
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # --- Optimizer and Scheduler Setup ---
    param_groups = None
    if 'galore' in args.optimizer_base or 'gale' in args.optimizer_base:
        method_name = "GaLore" if 'galore' in args.optimizer_base else "GALE"
        target_modules = ["attn", "mlp"] if "llama" in args.model_name_or_path.lower() else ["attention", "intermediate", "output"]
        
        low_rank_params = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                logger.info(f"Enabling {method_name} for weights in module: {name}")
                low_rank_params.append(module.weight)

        id_low_rank_params = {id(p) for p in low_rank_params}
        regular_params = [p for p in model.parameters() if id(p) not in id_low_rank_params and p.requires_grad]
        low_rank_pg = {'params': low_rank_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale}
        if 'galore' in args.optimizer_base:
            low_rank_pg['proj_type'] = args.proj_type
        param_groups = [{'params': regular_params}, low_rank_pg]
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    optimizer_classes = {
        'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'adam8bit': bnb.optim.Adam8bit,
        'adafactor': transformers.optimization.Adafactor, 'galore_adamw': GaLoreAdamW,
        'galore_adamw8bit': GaLoreAdamW8bit, 'galore_adafactor': GaLoreAdafactor,
        'swift_galore_adamw': SwiftGaLoreAdamW,
        'gale_adamw': GALE_AdamW, 'gale_adamw_fused': GALE_AdamW, 'gale_adamw_fused_approx': GALE_AdamW,
        'gale_adafactor': GALE_Adafactor, 'gale_adafactor_fused': GALE_Adafactor, 'gale_adafactor_fused_approx': GALE_Adafactor,
        'gale_adamw8bit': GALE_AdamW8bit, 'gale_adamw8bit_fused': GALE_AdamW8bit, 'gale_adamw8bit_fused_approx': GALE_AdamW8bit,
        'lion': Lion, 'gale_lion': GALE_Lion
    }
    optimizer_class = optimizer_classes[args.optimizer_base]
    optimizer_kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    
    if args.optimizer_base in ['adafactor', 'galore_adafactor']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False})
    elif args.optimizer_base in ['gale_adamw']:
        optimizer_kwargs['mode'] = 'native'
    elif args.optimizer_base in ['gale_adamw_fused']:
        optimizer_kwargs['mode'] = 'fused'
    elif args.optimizer_base in ['gale_adamw_fused_approx']:
        optimizer_kwargs['mode'] = 'approximate'
    elif args.optimizer_base in ['gale_adafactor']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'native'})
    elif args.optimizer_base in ['gale_adafactor_fused']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'fused'})
    elif args.optimizer_base in ['gale_adafactor_fused_approx']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'approximate'})
    elif args.optimizer_base in ['gale_adamw8bit']:
        optimizer_kwargs['mode'] = 'native'
    elif args.optimizer_base in ['gale_adamw8bit_fused']:
        optimizer_kwargs['mode'] = 'fused'
    elif args.optimizer_base in ['gale_adamw8bit_fused_approx']:
        optimizer_kwargs['mode'] = 'approximate'
    
    optimizer = optimizer_class(param_groups, **optimizer_kwargs)
    
    # Calculate theoretical memory AFTER optimizer setup
    theoretical_mem_mib = calculate_theoretical_memory(model, args)
    logger.info(f"[seed {seed}] Theoretical Memory (BF16): {theoretical_mem_mib:.2f} MiB")
    
    # --- Training Loop Setup ---
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps,
    )
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.task_name in ("boolq", "cb"):
        metric = evaluate.load("super_glue", args.task_name)
    elif args.task_name == "anli_r1":
        metric = evaluate.load("accuracy")
    else:
        metric = evaluate.load("glue", args.task_name)
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    
    logger.info(f"[seed {seed}] ***** Training *****")
    logger.info(f"[seed {seed}] Epochs={args.num_train_epochs} | Steps={args.max_train_steps} | Total batch={args.total_batch_size}")
    
    step_times: List[float] = []
    mem_stats_after_first_step = {}
    best_metric_val = float("-inf")
    best_metric_dict: Dict[str, float] = {}
    
    # --- Training Loop ---
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):

            # Move batch to device and cast to appropriate dtype
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }
            if is_regression and "labels" in batch:
                batch["labels"] = batch["labels"].to(dtype)

            outputs = model(**batch)
            loss = outputs.loss
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step == len(train_loader) - 1):
                if args.grad_clipping > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
                
                step_start_time = time.perf_counter()
                optimizer.step()
                step_times.append(time.perf_counter() - step_start_time)

                lr_scheduler.step()

                # AdaLoRA: update rank allocation BEFORE zero_grad (needs gradients)
                # Only run when actual pruning is configured (init_r > target_r)
                if args.adapter_method == 'adalora' and args.adalora_init_r > args.adalora_target_r:
                    model.base_model.update_and_allocate(completed_steps + 1)

                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps == 1 and device.type == "cuda":
                    torch.cuda.empty_cache()
                    mem_stats_after_first_step = get_memory_breakdown(model, optimizer, device)
                    logger.info(
                        "Memory breakdown after 1st optimizer step: | "
                        f"Param Memory: {mem_stats_after_first_step.get('param_mem_mib', 0):.2f} MiB | "
                        f"Optimizer Memory: {mem_stats_after_first_step.get('opt_mem_mib', 0):.2f} MiB | "
                        f"Allocated Memory: {mem_stats_after_first_step.get('allocated_memory_mib', 0):.2f} MiB | "
                        f"Peak Memory: {mem_stats_after_first_step.get('peak_memory_mib', 0):.2f} MiB"
                    )
            
            if completed_steps >= args.max_train_steps:
                break
        
        # --- Evaluation ---
        model.eval()
        for batch in eval_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            refs = batch["labels"]
            metric.add_batch(predictions=preds.cpu(), references=refs.cpu())

        eval_metric = metric.compute()
        logger.info(f"[seed {seed}] epoch {epoch}: {eval_metric}")

        primary_val = _primary_metric(args.task_name, eval_metric)
        if primary_val > best_metric_val:
            best_metric_val = primary_val
            best_metric_dict = eval_metric.copy()
        
        if completed_steps >= args.max_train_steps:
            break
            
    # --- Final Benchmarks ---
    peak_mem_mib = mem_stats_after_first_step.get('peak_memory_mib', 0)
    if device.type == "cuda":
        final_peak_memory_mib = mib(torch.cuda.max_memory_allocated(device))
        logger.info(f"[seed {seed}] Overall Peak GPU Memory (whole run): {final_peak_memory_mib:.2f} MiB")
        peak_mem_mib = max(peak_mem_mib, final_peak_memory_mib)

    if not step_times:
        avg_step_time = std_step_time = np.nan
    else:
        avg_step_time = statistics.mean(step_times)
        std_step_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    # --- Cleanup ---
    del model, optimizer, train_loader, eval_loader, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "best_metric_dict": best_metric_dict,
        "param_mem_mib": mem_stats_after_first_step.get('param_mem_mib', 0),
        "opt_mem_mib": mem_stats_after_first_step.get('opt_mem_mib', 0),
        "runtime_mem_mib": mem_stats_after_first_step.get('allocated_memory_mib', 0),
        "peak_mem_mib": peak_mem_mib,
        "theoretical_mem_mib": theoretical_mem_mib,
        "avg_step_time": avg_step_time,
        "std_step_time": std_step_time,
    }

###############################################################################
#                                  entry-point                                #
###############################################################################
def main():
    args = parse_args()
    
    training_start_time = time.time()
    all_results: List[Dict] = []
    for idx, seed in enumerate(SEEDS):
        print("=" * 80, flush=True)
        print(f"Starting run {idx + 1}/{len(SEEDS)} with seed {seed}", flush=True)
        print("=" * 80, flush=True)
        res = run_single_seed(args, seed)
        all_results.append(res)
    
    total_training_time_sec = time.time() - training_start_time

    # --- Process and Save Results ---
    first_res = all_results[0]
    metric_keys = ["accuracy", "f1", "matthews_correlation", "pearson", "spearmanr"]
    median_metrics = {}
    for k in metric_keys:
        vals = [r["best_metric_dict"].get(k, np.nan) for r in all_results]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        median_metrics[k] = statistics.median(vals) if vals else np.nan

    all_columns = [
        "timestamp", "name", "model_name_or_path", "task_name", "optimizer",
        "lr", "per_device_train_batch_size", "total_batch_size", "num_train_epochs",
        "max_train_steps", "dtype", "adapter_method",
        "rank", "update_proj_gap", "galore_scale",
        "lora_r", "lora_alpha", "lora_dropout", "prefix_bottleneck_size",
        "dora_r", "dora_alpha", "dora_dropout",
        "vera_r", "vera_dropout", "vera_d_initial",
        "gbvera_r", "gbvera_d_initial", "gbvera_b_initial", "gbvera_dropout",
        "fourierft_n_frequency", "fourierft_scaling",
        "adalora_init_r", "adalora_target_r", "adalora_alpha", "adalora_dropout",
        "dylora_r", "dylora_alpha", "dylora_dropout",
        "spectral_p", "spectral_q", "spectral_scaling", "spectral_dropout", "spectral_d_initial", "spectral_freq_mode", "spectral_freq_exponent", "spectral_factored_rank", "spectral_learn_scaling",
        "per_layer_opt", "gradient_checkpointing", "accuracy", "f1", "matthews_correlation", "pearson", "spearmanr",
        "total_training_time_sec", "param_mem_mib", "opt_mem_mib", "runtime_mem_mib",
        "peak_mem_mib", "theoretical_mem_mib", "avg_step_time", "std_step_time", "seed"
    ]
    comb_cols = ["model_name_or_path", "task_name", "optimizer", "lr", "total_batch_size"]

    is_galore_or_gale = 'galore' in args.optimizer.lower() or 'gale' in args.optimizer.lower()
    
    result_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": args.name,
        "model_name_or_path": args.model_name_or_path,
        "task_name": args.task_name,
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "total_batch_size": args.total_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "dtype": args.dtype,
        "adapter_method": args.adapter_method if args.adapter_method else 'N/A',
        "rank": args.rank if is_galore_or_gale else 'N/A',
        "update_proj_gap": args.update_proj_gap if is_galore_or_gale else 'N/A',
        "galore_scale": args.galore_scale if is_galore_or_gale else 'N/A',
        "lora_r": args.lora_r if args.adapter_method == 'lora' else 'N/A',
        "lora_alpha": args.lora_alpha if args.adapter_method == 'lora' else 'N/A',
        "lora_dropout": args.lora_dropout if args.adapter_method == 'lora' else 'N/A',
        "prefix_bottleneck_size": args.prefix_bottleneck_size if args.adapter_method == 'prefix' else 'N/A',
        "dora_r": args.dora_r if args.adapter_method == 'dora' else 'N/A',
        "dora_alpha": args.dora_alpha if args.adapter_method == 'dora' else 'N/A',
        "dora_dropout": args.dora_dropout if args.adapter_method == 'dora' else 'N/A',
        "vera_r": args.vera_r if args.adapter_method == 'vera' else 'N/A',
        "vera_dropout": args.vera_dropout if args.adapter_method == 'vera' else 'N/A',
        "vera_d_initial": args.vera_d_initial if args.adapter_method == 'vera' else 'N/A',
        "gbvera_r": args.gbvera_r if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_d_initial": args.gbvera_d_initial if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_b_initial": args.gbvera_b_initial if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_dropout": args.gbvera_dropout if args.adapter_method == 'gbvera' else 'N/A',
        "fourierft_n_frequency": args.fourierft_n_frequency if args.adapter_method == 'fourierft' else 'N/A',
        "fourierft_scaling": args.fourierft_scaling if args.adapter_method == 'fourierft' else 'N/A',
        "adalora_init_r": args.adalora_init_r if args.adapter_method == 'adalora' else 'N/A',
        "adalora_target_r": args.adalora_target_r if args.adapter_method == 'adalora' else 'N/A',
        "adalora_alpha": args.adalora_alpha if args.adapter_method == 'adalora' else 'N/A',
        "adalora_dropout": args.adalora_dropout if args.adapter_method == 'adalora' else 'N/A',
        "dylora_r": args.dylora_r if args.adapter_method == 'dylora' else 'N/A',
        "dylora_alpha": args.dylora_alpha if args.adapter_method == 'dylora' else 'N/A',
        "dylora_dropout": args.dylora_dropout if args.adapter_method == 'dylora' else 'N/A',
        "spectral_p": args.spectral_p if args.adapter_method == 'spectral' else 'N/A',
        "spectral_q": args.spectral_q if args.adapter_method == 'spectral' else 'N/A',
        "spectral_scaling": args.spectral_scaling if args.adapter_method == 'spectral' else 'N/A',
        "spectral_dropout": args.spectral_dropout if args.adapter_method == 'spectral' else 'N/A',
        "spectral_d_initial": args.spectral_d_initial if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_mode": args.spectral_freq_mode if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_exponent": args.spectral_freq_exponent if args.adapter_method == 'spectral' else 'N/A',
        "spectral_factored_rank": args.spectral_factored_rank if args.adapter_method == 'spectral' else 'N/A',
        "spectral_learn_scaling": args.spectral_learn_scaling if args.adapter_method == 'spectral' else 'N/A',
        "per_layer_opt": args.per_layer_opt,
        "gradient_checkpointing": args.gradient_checkpointing,
        "accuracy": median_metrics.get("accuracy", np.nan),
        "f1": median_metrics.get("f1", np.nan),
        "matthews_correlation": median_metrics.get("matthews_correlation", np.nan),
        "pearson": median_metrics.get("pearson", np.nan),
        "spearmanr": median_metrics.get("spearmanr", np.nan),
        "total_training_time_sec": round(total_training_time_sec, 2),
        "param_mem_mib": round(first_res["param_mem_mib"], 2),
        "opt_mem_mib": round(first_res["opt_mem_mib"], 2),
        "runtime_mem_mib": round(first_res["runtime_mem_mib"], 2),
        "peak_mem_mib": round(first_res["peak_mem_mib"], 2),
        "theoretical_mem_mib": round(first_res["theoretical_mem_mib"], 2),
        "avg_step_time": round(first_res["avg_step_time"], 4) if first_res["avg_step_time"] is not np.nan else np.nan,
        "std_step_time": round(first_res["std_step_time"], 4) if first_res["std_step_time"] is not np.nan else np.nan,
        "seed": ",".join(map(str, SEEDS)),
    }
    # --- This is the new locking mechanism ---
    # Create a lock object. Timeout is optional but good practice.
    lock = FileLock(LOCK_FILE_PATH, timeout=60)

    with lock:
        logger.info(f"Acquired lock on {LOCK_FILE_PATH} to update results.")
        df_results = _load_results_df(all_columns)
        df_results = _upsert_result(df_results, comb_cols, result_row)
        df_results.to_csv(RESULTS_FILE, index=False)
        logger.info(f"Released lock. Logged Mo5 median results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
