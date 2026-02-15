#!/bin/bash
# ============================================================================
# PEFT Comprehensive Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits Mo5 (median-of-five, seeds 41-45) benchmarks for PEFT methods on GLUE tasks.
# Each job runs train_glue.py which trains 5 seeds and writes the median
# results to ./results/mo53_glue.csv with file locking for concurrent safety.
#
# MODULE TARGETING: All methods target Q+V (query, value) = 24 modules on BERT-base.
# Each method uses its natural minimum config — no parameter matching.
#
# Techniques and approximate param counts on BERT-base (Q+V, 24 modules):
#   - base:         Full fine-tuning (~110M)        Performance ceiling
#   - lora:         LoRA r=1, Q+V (~38K)            Minimum rank (24 modules)
#   - dora:         DoRA r=1, Q+V (~57K)            Minimum rank + magnitude vectors
#   - adalora:      AdaLoRA r=1, Q+V (~38K)         Minimum rank; = LoRA at r=1
#   - dylora:       DyLoRA r=1, Q+V (~38K)          Minimum rank; = LoRA at r=1
#   - vera:         VeRA r=128, Q+V (~8K)           Natural config (24 modules)
#   - fourierft:    FourierFT n=256, Q+V (~8K)      Param-matched with Spectral p=16
#   - spectral_p16: Spectral Adapter p=q=32,r=4 (~8K) Ours (factored+learn_scale)
#
# At r=1, DyLoRA and AdaLoRA are functionally identical to LoRA (no rank sampling,
# no pruning when init_r=target_r).
#
# Usage:
#   ./sbatch/run_peft_experiments.sh
#   ./sbatch/run_peft_experiments.sh --account def-myprof
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION - Modify these arrays to control what runs
# ============================================================================

# Models to benchmark
models=(
    "bert-base-uncased"
    #"gpt2"
    # "roberta-base"
    # "bert-large-uncased"
)

# Techniques to benchmark
# All PEFT methods target Q+V (24 modules) at minimum rank/config.
techniques=(
    #"base"
    #"lora"
    #"dora"
    #"adalora"
    #"dylora"
    #"vera"
    #"fourierft"
    "spectral_p16"
    # --- Higher-budget variants (uncomment for extended comparison) ---
    # "lora_r8"        # LoRA r=8 standard config (~295K params)
    # "dora_r16"       # DoRA r=16 standard config (~600K+ params)
    # "vera_r256"      # VeRA r=256 standard config (~48K params)
    # "fourierft_n1k"  # FourierFT n=1000 standard config (~75K params)
    # "spectral_p32"   # Spectral p=q=32 (~76K params)
)

# GLUE tasks to evaluate
tasks=(
    "mrpc"
    "sst2"
    "cola"
    "rte"
    "qnli"
    "stsb"
    #"mnli"     # 393K train - very expensive, uncomment if needed
    #"qqp"      # 364K train - very expensive, uncomment if needed
    "boolq"    # 9.4K train (SuperGLUE)
    "cb"       # 250 train (SuperGLUE)
    "anli_r1"  # 16.9K train (Adversarial NLI Round 1)
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
#
# All methods target Q+V = 24 modules of 768x768 on BERT-base.
# Classifier head: 768x2 + 2 = 1,538 always-trainable params.
# Each method uses its minimum rank/config — no parameter matching.
#
# ============================================================================

# --- Shared across all techniques ---
BATCH_SIZE=32
EVAL_BATCH_SIZE=32
WEIGHT_DECAY=0.01
LR_SCHEDULER="linear"
DTYPE="float32"       # Uniform dtype for fair comparison (required for spectral/FourierFT)
GRAD_CLIP=1.0

# --- Full fine-tuning (Devlin et al., 2019) ---
#     ~110M params (performance ceiling)
BASE_LR="2e-5"
BASE_EPOCHS=3

# --- LoRA (Hu et al., 2021) via AdapterHub ---
#     r=1, Q+V (24 modules of 768x768).
#     Params: 24 x 1536 x 1 + 1,538 classifier = 38,402
#     alpha/r ratio kept at 2 (matching standard alpha=16/r=8 convention)
LORA_LR="2e-4"
LORA_R=1
LORA_ALPHA=2
LORA_DROPOUT=0.0

# --- DoRA (Liu et al., 2024) via PEFT ---
#     r=1, Q+V (24 modules of 768x768).
#     LoRA part: 24 x 1536 x 1 = 36,864
#     Magnitude vectors: 24 x 768 = 18,432
#     Total: 36,864 + 18,432 + 1,538 = 56,834
DORA_LR="2e-4"
DORA_R=1
DORA_ALPHA=2
DORA_DROPOUT=0.05

# --- VeRA (Kopiczko et al., 2024) via PEFT ---
#     Shared frozen random projections; only per-module scaling vectors are trainable.
#     Trainable: 24 modules x 2r (lambda_d + lambda_b) = 48r + 1,538 classifier
#     r=128 -> 6,144 + 1,538 = 7,682 params
VERA_LR="2e-3"
VERA_R=128
VERA_D_INITIAL=0.1
VERA_DROPOUT=0.0

# --- FourierFT (Gao et al., ICML 2024) via PEFT ---
#     n spectral coefficients per module, Q+V (24 modules of 768x768).
#     Trainable: 24 modules x n + 1,538 classifier
#     n=256 -> 6,144 + 1,538 = 7,682 params (param-matched with Spectral p=16)
#     scaling 100-150 for GLUE (PEFT docs); lr=5e-2 validated by sanity check
FOURIERFT_LR="5e-2"
FOURIERFT_N=256
FOURIERFT_SCALING=150.0

# --- Spectral Adapter (ours) ---
#     Q+V (24 modules of 768x768).
#     Factored S: p=q=32, rank=4 → 32*4 + 4*32 = 256 params/module
#     + per-module learnable scaling (+1 param/module)
#     Trainable: 24 modules x 257 + 1,538 classifier = 7,706 params
SPECTRAL_LR="2e-2"
SPECTRAL_P=32
SPECTRAL_Q=32
SPECTRAL_SCALING=1.0
SPECTRAL_DROPOUT=0.0
SPECTRAL_D_INITIAL=0.01  # Nonzero init fixes CoLA underperformance (late adapter onset)
SPECTRAL_FACTORED_RANK=4
SPECTRAL_LEARN_SCALING=true  # Per-module learnable scaling (+24 params total)

# --- AdaLoRA (Zhang et al., 2023) via PEFT ---
#     SVD-parameterized LoRA with adaptive rank allocation.
#     r=1, Q+V (24 modules of 768x768).
#     Params: 24 x (768 + 768 + 1) + 1,538 = 38,426
#     NOTE: init_r=target_r=1 → no pruning → = LoRA.
ADALORA_LR="2e-3"
ADALORA_INIT_R=1
ADALORA_TARGET_R=1
ADALORA_ALPHA=2

# --- DyLoRA (Valipour et al., EACL 2023) custom ---
#     Trains across randomly sampled ranks. At r=1, = standard LoRA.
#     r=1, Q+V (24 modules of 768x768).
#     Params: 24 x (768 + 768) x 1 + 1,538 = 38,402
#     NOTE: r=1 → no rank sampling → = LoRA.
DYLORA_LR="2e-4"
DYLORA_R=1
DYLORA_ALPHA=2

# --- Q+V target modules for all PEFT methods on BERT ---
#     24 modules of 768x768 (12 layers x {query, value})
BERT_TARGET_MODULES="query,value"

# --- Higher-budget variant hyperparameters (for extended comparison) ---
LORA_R8_LR="2e-4"
LORA_R8_R=8
LORA_R8_ALPHA=16
LORA_R8_DROPOUT=0.0

DORA_R16_LR="2e-4"
DORA_R16_R=16
DORA_R16_ALPHA=32
DORA_R16_DROPOUT=0.05

VERA_R256_LR="2e-3"
VERA_R256_R=256
VERA_R256_D_INITIAL=0.1
VERA_R256_DROPOUT=0.0

FOURIERFT_N1K_LR="2e-3"
FOURIERFT_N1K_N=1000
FOURIERFT_N1K_SCALING=300.0

SPECTRAL_P32_LR="2e-3"
SPECTRAL_P32_SCALING=1.0
SPECTRAL_P32_DROPOUT=0.0

# --- Epochs: base=3 always; PEFT=30 (small tasks) / 10 (large tasks) ---
PEFT_EPOCHS_SMALL=30   # mrpc, rte, cola, stsb, wnli (<10K samples)
PEFT_EPOCHS_LARGE=10   # sst2, qnli, qqp, mnli (>=10K samples)

# ============================================================================
# END CONFIGURATION
# ============================================================================

job_count=0
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_config() {
    # Sets: gpu_type, mem
    local model=$1
    case $model in
        "bert-base-uncased"|"roberta-base"|"gpt2")
            gpu_type="nvidia_h100_80gb_hbm3_2g.20gb:1"
            mem="20000M"
            ;;
        "bert-large-uncased"|"roberta-large")
            gpu_type="nvidia_h100_80gb_hbm3_4g.40gb:1"
            mem="40000M"
            ;;
        *)
            # Default for unknown models: full 80GB GPU
            gpu_type="nvidia_h100_80gb_hbm3:1"
            mem="64000M"
            ;;
    esac
}

is_large_task() {
    # Large tasks (>=60K samples): 10 epochs
    # Small tasks (<20K samples): 30 epochs
    # BoolQ (9.4K), ANLI R1 (16.9K), CB (250) are all small
    case $1 in
        sst2|qnli|qqp|mnli) return 0 ;;
        *) return 1 ;;
    esac
}

get_epochs() {
    # Returns epoch count based on technique and task size
    local technique=$1
    local task=$2
    if [[ "$technique" == "base" ]]; then
        echo "$BASE_EPOCHS"
    elif is_large_task "$task"; then
        echo "$PEFT_EPOCHS_LARGE"
    else
        echo "$PEFT_EPOCHS_SMALL"
    fi
}

get_time_limit() {
    # Returns SLURM time string based on technique, task, and model
    # Mo5 (5 seeds): 5x single-seed estimates with ~2x safety margin each.
    local technique=$1
    local task=$2
    local model=$3
    local minutes=0

    if [[ "$technique" == "base" ]]; then
        # Full FT: 3 epochs, 5 seeds
        case $task in
            mrpc|rte|stsb|wnli|cb) minutes=50   ;;
            cola)                   minutes=75   ;;
            boolq)                  minutes=100  ;;   # 9.4K train, 3ep
            sst2)                   minutes=150  ;;
            anli_r1)                minutes=200  ;;   # 16.9K train, 3ep
            qnli)                   minutes=275  ;;
            qqp|mnli)               minutes=600  ;;
        esac
    else
        # PEFT: frozen backbone, more epochs, 5 seeds
        case $task in
            rte|wnli|cb) minutes=150  ;;   # 2h30   (tiny dataset, 30ep)
            mrpc|stsb)   minutes=175  ;;   # 2h55   (small dataset, 30ep)
            cola)        minutes=275  ;;   # 4h35   (medium dataset, 30ep)
            boolq)       minutes=600  ;;   # 10h    (medium dataset, 9.4K, 30ep)
            anli_r1)     minutes=900  ;;   # 15h    (medium-large dataset, 16.9K, 30ep)
            sst2)        minutes=1200 ;;   # 20h    (large dataset, 10ep)
            qnli)        minutes=1800 ;;   # 30h    (xlarge dataset, 10ep)
            qqp|mnli)    minutes=3000 ;;   # 50h    (xxlarge dataset, 10ep)
        esac
    fi

    # Scale for larger models
    case $model in
        "bert-large-uncased"|"roberta-large") minutes=$((minutes * 3)) ;;
    esac

    # Format as D-HH:MM:SS or H:MM:SS
    local hours=$((minutes / 60))
    local mins=$((minutes % 60))
    if [[ $hours -ge 24 ]]; then
        local days=$((hours / 24))
        hours=$((hours % 24))
        printf "%d-%02d:%02d:00" "$days" "$hours" "$mins"
    else
        printf "%d:%02d:00" "$hours" "$mins"
    fi
}

build_python_cmd() {
    # Returns the full python command for a given (model, technique, task)
    local model=$1
    local technique=$2
    local task=$3
    local epochs=$4
    local run_name=$5

    local common="python src/train_glue.py"
    common+=" --model_name_or_path $model"
    common+=" --task_name $task"
    common+=" --per_device_train_batch_size $BATCH_SIZE"
    common+=" --per_device_eval_batch_size $EVAL_BATCH_SIZE"
    common+=" --num_train_epochs $epochs"
    common+=" --weight_decay $WEIGHT_DECAY"
    common+=" --lr_scheduler_type $LR_SCHEDULER"
    common+=" --grad_clipping $GRAD_CLIP"
    common+=" --dtype $DTYPE"
    common+=" --name $run_name"

    case $technique in
        base)
            echo "$common --optimizer adamw --learning_rate $BASE_LR"
            ;;
        lora)
            local cmd="$common --optimizer adamw-lora --learning_rate $LORA_LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        dora)
            local cmd="$common --optimizer adamw-dora --learning_rate $DORA_LR --dora_r $DORA_R --dora_alpha $DORA_ALPHA --dora_dropout $DORA_DROPOUT"
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        vera)
            local cmd="$common --optimizer adamw-vera --learning_rate $VERA_LR --vera_r $VERA_R --vera_d_initial $VERA_D_INITIAL --vera_dropout $VERA_DROPOUT"
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        fourierft)
            local cmd="$common --optimizer adamw-fourierft --learning_rate $FOURIERFT_LR --fourierft_n_frequency $FOURIERFT_N --fourierft_scaling $FOURIERFT_SCALING"
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        adalora)
            local cmd="$common --optimizer adamw-adalora --learning_rate $ADALORA_LR --adalora_init_r $ADALORA_INIT_R --adalora_target_r $ADALORA_TARGET_R --adalora_alpha $ADALORA_ALPHA"
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        dylora)
            local cmd="$common --optimizer adamw-dylora --learning_rate $DYLORA_LR --dylora_r $DYLORA_R --dylora_alpha $DYLORA_ALPHA"
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        spectral_p16)
            local cmd="$common --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --spectral_p $SPECTRAL_P --spectral_q $SPECTRAL_Q --spectral_scaling $SPECTRAL_SCALING --spectral_dropout $SPECTRAL_DROPOUT --spectral_d_initial $SPECTRAL_D_INITIAL --spectral_factored_rank $SPECTRAL_FACTORED_RANK"
            if [[ "$SPECTRAL_LEARN_SCALING" == "true" ]]; then
                cmd+=" --spectral_learn_scaling"
            fi
            if [[ "$model" == *"bert"* ]]; then
                cmd+=" --adapter_target_modules $BERT_TARGET_MODULES"
            fi
            echo "$cmd"
            ;;
        # --- Higher-budget variants ---
        lora_r8)
            echo "$common --optimizer adamw-lora --learning_rate $LORA_R8_LR --lora_r $LORA_R8_R --lora_alpha $LORA_R8_ALPHA --lora_dropout $LORA_R8_DROPOUT"
            ;;
        dora_r16)
            echo "$common --optimizer adamw-dora --learning_rate $DORA_R16_LR --dora_r $DORA_R16_R --dora_alpha $DORA_R16_ALPHA --dora_dropout $DORA_R16_DROPOUT"
            ;;
        vera_r256)
            echo "$common --optimizer adamw-vera --learning_rate $VERA_R256_LR --vera_r $VERA_R256_R --vera_d_initial $VERA_R256_D_INITIAL --vera_dropout $VERA_R256_DROPOUT"
            ;;
        fourierft_n1k)
            echo "$common --optimizer adamw-fourierft --learning_rate $FOURIERFT_N1K_LR --fourierft_n_frequency $FOURIERFT_N1K_N --fourierft_scaling $FOURIERFT_N1K_SCALING"
            ;;
        spectral_p32)
            echo "$common --optimizer adamw-spectral --learning_rate $SPECTRAL_P32_LR --spectral_p 32 --spectral_q 32 --spectral_scaling $SPECTRAL_P32_SCALING --spectral_dropout $SPECTRAL_P32_DROPOUT --spectral_d_initial $SPECTRAL_D_INITIAL --spectral_learn_scaling"
            ;;
    esac
}

get_technique_desc() {
    # Returns a short description for logging
    case $1 in
        base)           echo "Full FT (lr=$BASE_LR, ${BASE_EPOCHS}ep, ~110M params)" ;;
        lora)           echo "LoRA (r=$LORA_R, a=$LORA_ALPHA, Q+V, lr=$LORA_LR, ~38K params)" ;;
        dora)           echo "DoRA (r=$DORA_R, a=$DORA_ALPHA, Q+V, lr=$DORA_LR, ~57K params)" ;;
        adalora)        echo "AdaLoRA (init_r=$ADALORA_INIT_R, target_r=$ADALORA_TARGET_R, Q+V, lr=$ADALORA_LR, ~38K params)" ;;
        dylora)         echo "DyLoRA (r=$DYLORA_R, a=$DYLORA_ALPHA, Q+V, lr=$DYLORA_LR, ~38K params)" ;;
        vera)           echo "VeRA (r=$VERA_R, Q+V, lr=$VERA_LR, ~8K params)" ;;
        fourierft)      echo "FourierFT (n=$FOURIERFT_N, s=$FOURIERFT_SCALING, Q+V, lr=$FOURIERFT_LR, ~8K params)" ;;
        spectral_p16)   echo "Spectral (p=$SPECTRAL_P, q=$SPECTRAL_Q, r=$SPECTRAL_FACTORED_RANK, learn_scale, Q+V, lr=$SPECTRAL_LR, ~8K params)" ;;
        lora_r8)        echo "LoRA (r=$LORA_R8_R, a=$LORA_R8_ALPHA, lr=$LORA_R8_LR, ~295K params)" ;;
        dora_r16)       echo "DoRA (r=$DORA_R16_R, a=$DORA_R16_ALPHA, lr=$DORA_R16_LR, ~600K params)" ;;
        vera_r256)      echo "VeRA (r=$VERA_R256_R, lr=$VERA_R256_LR, ~48K params)" ;;
        fourierft_n1k)  echo "FourierFT (n=$FOURIERFT_N1K_N, s=$FOURIERFT_N1K_SCALING, lr=$FOURIERFT_N1K_LR, ~75K params)" ;;
        spectral_p32)   echo "Spectral (p=32, q=32, lr=$SPECTRAL_P32_LR, ~76K params)" ;;
    esac
}

# ============================================================================
# MAIN LOOP - Submit one job per (model, technique, task)
# ============================================================================

echo "============================================"
echo "PEFT Benchmark Suite - Job Submission"
echo "============================================"
echo "Models:     ${models[*]}"
echo "Techniques: ${techniques[*]}"
echo "Tasks:      ${tasks[*]}"
echo "Dtype:      $DTYPE"
echo "Target:     Q+V (24 modules), min config per method"
echo "============================================"
echo ""

for model in "${models[@]}"; do
    # Short model name for job IDs (e.g., bert-base-uncased -> bert-base)
    model_short=$(basename "$model" | sed 's/-uncased//; s/-cased//')

    get_job_config "$model"

    for technique in "${techniques[@]}"; do
        technique_desc=$(get_technique_desc "$technique")

        for task in "${tasks[@]}"; do
            epochs=$(get_epochs "$technique" "$task")
            time_limit=$(get_time_limit "$technique" "$task" "$model")
            job_name="peft_${model_short}_${technique}_${task}"
            run_name="${technique}_${model_short}_${task}_${DTYPE}"

            python_cmd=$(build_python_cmd "$model" "$technique" "$task" "$epochs" "$run_name")

            account_line=""
            if [[ -n "$ACCOUNT" ]]; then
                account_line="#SBATCH --account=$ACCOUNT"
            fi

            sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH --gres=gpu:$gpu_type
#SBATCH --mem=$mem
#SBATCH --cpus-per-task=4
$account_line

            module load gcc arrow scipy-stack cuda cudnn
            source ./env/bin/activate

            export HF_HOME=\$(pwd)/data
            export HF_DATASETS_CACHE=\$(pwd)/data
            export TRANSFORMERS_CACHE=\$(pwd)/data
            export TORCH_HOME=\$(pwd)/data
            mkdir -p \$HF_HOME

            echo '========================================'
            echo 'Job: $job_name'
            echo 'Model: $model'
            echo 'Technique: $technique'
            echo 'Config: $technique_desc'
            echo 'Task: $task'
            echo 'Epochs: $epochs'
            echo 'Dtype: $DTYPE'
            echo 'Time limit: $time_limit'
            echo 'Cache: '\$HF_HOME
            echo 'Started: '\$(date)
            echo '========================================'
            nvidia-smi
            export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src
            python -c "import sys; print('\n'.join(sys.path))"
            python -c "import pandas; print(pandas.__file__)" 2>&1 || echo "pandas NOT FOUND"
            $python_cmd
            echo '========================================'
            echo 'Finished: '\$(date)
            echo '========================================'
EOF
)
            echo "  [$sbatch_id] $job_name  ($technique_desc, ${task}, ${epochs}ep, ${time_limit})"
            ((job_count++))
        done
    done
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          ./results/mo53_glue.csv"
echo "Logs directory:       ./logs/"
echo "============================================"
