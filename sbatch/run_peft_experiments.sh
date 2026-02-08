#!/bin/bash
# ============================================================================
# PEFT Comprehensive Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits Mo5 (median-of-5-seeds) benchmarks for PEFT methods on GLUE tasks.
# Each job runs train_glue.py which internally loops seeds 41-45 and writes
# results to ./results/mo53_glue.csv with file locking for concurrent safety.
#
# PARAMETER BUDGET: ~20K trainable params (matching Spectral p=16 best result)
#
# Techniques and approximate param counts on BERT-base:
#   - base:         Full fine-tuning (~110M)        Performance ceiling
#   - lora:         LoRA r=1 (~38K)                 Structural minimum; cannot reach 20K
#   - dora:         DoRA r=1 (~253K)                Structural minimum; cannot reach 20K
#   - vera:         VeRA r=128 (~20K)               Parameter-matched
#   - fourierft:    FourierFT n=252 (~20K)          Parameter-matched
#   - spectral_p16: Spectral Adapter p=q=16 (~20K)  Ours (best result)
#
# LoRA/DoRA cannot reach the 20K budget even at r=1 due to their per-module
# parameter structure. They are included at their structural minimum (r=1)
# to demonstrate that Spectral achieves better performance with fewer params.
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
    # "roberta-base"
    # "bert-large-uncased"
)

# Techniques to benchmark
# All PEFT methods tuned to ~20K params where structurally possible.
# LoRA/DoRA included at structural minimum (r=1) for completeness.
techniques=(
    "base"
    #"lora"
    #"dora"
    #"vera"
    #"fourierft"
    #"spectral_p16"
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
    #"sst2"
    #"cola"
    #"rte"
    #"qnli"
    #"stsb"
    #"mnli"   # 393K train - very expensive, uncomment if needed
    # "qqp"    # 364K train - very expensive, uncomment if needed
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
#
# Target budget: ~20K trainable params (Spectral p=16 = 20,226 params)
#
# BERT-base module counts for target_modules=["query","key","value","dense"]:
#   49 modules at 768x768  (Q, K, V, attn.output.dense, pooler.dense)
#   12 modules at 3072x768 (intermediate.dense)
#   12 modules at 768x3072 (output.dense)
#   = 73 modules total
#   + classifier head: 768x2 + 2 = 1,538 always-trainable params
#
# AdapterHub LoRA default on BERT: query+value only = 24 modules (768x768)
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
#     r=1 is the structural minimum. AdapterHub targets Q+V (24 modules).
#     Params: 24 modules x 1536 x r = 36,864r + 1,538 classifier
#     r=1 -> ~38K params (CANNOT reach 20K; minimum viable config)
#     alpha/r ratio kept at 2 (matching standard alpha=16/r=8 convention)
LORA_LR="2e-4"
LORA_R=1
LORA_ALPHA=2
LORA_DROPOUT=0.0

# --- DoRA (Liu et al., 2024) via PEFT ---
#     r=1 is the structural minimum. Targets Q+K+V+Dense (73 modules).
#     LoRA part: 49x1536 + 12x3840 + 12x3840 = 167,424 at r=1
#     Magnitude vectors: 49x768 + 12x3072 + 12x768 = 83,712
#     r=1 -> ~253K params (CANNOT reach 20K; structural floor is ~85K from magnitudes alone)
DORA_LR="2e-4"
DORA_R=1
DORA_ALPHA=2
DORA_DROPOUT=0.05

# --- VeRA (Kopiczko et al., 2024) via PEFT ---
#     Shared frozen random projections; only per-module scaling vectors are trainable.
#     Trainable: 73 modules x 2r (lambda_d + lambda_b) = 146r + 1,538 classifier
#     r=128 -> 18,688 + 1,538 = ~20K params  (MATCHED)
VERA_LR="2e-3"
VERA_R=128
VERA_D_INITIAL=0.1
VERA_DROPOUT=0.0

# --- FourierFT (Gao et al., ICML 2024) via PEFT ---
#     n spectral coefficients per module.
#     Trainable: 73 modules x n + 1,538 classifier
#     n=252 -> 18,396 + 1,538 = ~20K params  (MATCHED)
#     scaling=300 is the paper's default for GLUE sequence classification
FOURIERFT_LR="2e-3"
FOURIERFT_N=252
FOURIERFT_SCALING=300.0

# --- Spectral Adapter p=16 (ours) ---
#     Trainable: 73 modules x (16 x 16) + 1,538 classifier
#     = 18,688 + 1,538 = 20,226 params  (REFERENCE)
SPECTRAL_LR="2e-3"
SPECTRAL_SCALING=1.0
SPECTRAL_DROPOUT=0.0

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
        "bert-base-uncased"|"roberta-base")
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
    # Estimates derived from actual BERT-base timings:
    #   FourierFT MRPC Mo5 (30ep): 3405s ~57min
    #   Spectral  MRPC Mo5 (30ep): 1785s ~30min
    #   FourierFT SST2 Mo5 (10ep): 42715s ~12h
    #   Spectral  SST2 Mo5 (10ep): 21309s ~6h
    # Time limits include ~2x safety margin.
    local technique=$1
    local task=$2
    local model=$3
    local minutes=0

    if [[ "$technique" == "base" ]]; then
        # Full FT: 3 epochs, all params update but very few epochs
        case $task in
            mrpc|rte|stsb|wnli)  minutes=30   ;;
            cola)                 minutes=45   ;;
            sst2)                 minutes=120  ;;
            qnli)                 minutes=240  ;;
            qqp|mnli)             minutes=480  ;;
        esac
    else
        # PEFT: frozen backbone, more epochs
        case $task in
            rte|wnli)   minutes=120  ;;   # 2h    (tiny dataset, 30ep)
            mrpc|stsb)  minutes=150  ;;   # 2.5h  (small dataset, 30ep)
            cola)       minutes=240  ;;   # 4h    (medium dataset, 30ep)
            sst2)       minutes=1080 ;;   # 18h   (large dataset, 10ep)
            qnli)       minutes=1680 ;;   # 28h   (xlarge dataset, 10ep)
            qqp|mnli)   minutes=2880 ;;   # 48h   (xxlarge dataset, 10ep)
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
            echo "$common --optimizer adamw-lora --learning_rate $LORA_LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
            ;;
        dora)
            echo "$common --optimizer adamw-dora --learning_rate $DORA_LR --dora_r $DORA_R --dora_alpha $DORA_ALPHA --dora_dropout $DORA_DROPOUT"
            ;;
        vera)
            echo "$common --optimizer adamw-vera --learning_rate $VERA_LR --vera_r $VERA_R --vera_d_initial $VERA_D_INITIAL --vera_dropout $VERA_DROPOUT"
            ;;
        fourierft)
            echo "$common --optimizer adamw-fourierft --learning_rate $FOURIERFT_LR --fourierft_n_frequency $FOURIERFT_N --fourierft_scaling $FOURIERFT_SCALING"
            ;;
        spectral_p16)
            echo "$common --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --spectral_p 16 --spectral_q 16 --spectral_scaling $SPECTRAL_SCALING --spectral_dropout $SPECTRAL_DROPOUT"
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
            echo "$common --optimizer adamw-spectral --learning_rate $SPECTRAL_P32_LR --spectral_p 32 --spectral_q 32 --spectral_scaling $SPECTRAL_P32_SCALING --spectral_dropout $SPECTRAL_P32_DROPOUT"
            ;;
    esac
}

get_technique_desc() {
    # Returns a short description for logging
    case $1 in
        base)           echo "Full FT (lr=$BASE_LR, ${BASE_EPOCHS}ep, ~110M params)" ;;
        lora)           echo "LoRA (r=$LORA_R, a=$LORA_ALPHA, lr=$LORA_LR, ~38K params)" ;;
        dora)           echo "DoRA (r=$DORA_R, a=$DORA_ALPHA, lr=$DORA_LR, ~253K params)" ;;
        vera)           echo "VeRA (r=$VERA_R, lr=$VERA_LR, ~20K params)" ;;
        fourierft)      echo "FourierFT (n=$FOURIERFT_N, s=$FOURIERFT_SCALING, lr=$FOURIERFT_LR, ~20K params)" ;;
        spectral_p16)   echo "Spectral (p=16, q=16, lr=$SPECTRAL_LR, ~20K params)" ;;
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
echo "Target:     ~20K params (Spectral p=16)"
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
            export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)/src\"
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
