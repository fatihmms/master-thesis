#!/bin/bash
#SBATCH --job-name=kle_nqopen_llama
#SBATCH --output=logs/nqopen_llama_%A_%a.out    # %A = array job id, %a = task id (0-7)
#SBATCH --error=logs/nqopen_llama_%A_%a.err
#SBATCH --time=24:00:00                          # <= 48h so it stays valid on gpu_a100_il too (h100 max 72h)
#SBATCH --partition=gpu_h100                     # ACCOUNT A lane. Override at submit: sbatch --partition=gpu_a100_il ...
#SBATCH --gres=gpu:1                             # 1 GPU per array task (8B bf16 + DeBERTa fits everywhere)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7                              # 0-3 -> black-box c1-c4, 4-7 -> white-box c1-c4

# =====================================================================
# NQ-Open  x  Llama-3.1-8B-Instruct   (--model 8b)
# Submit FROM the repo root:  cd <repo> && sbatch scripts/nqopen_llama.sh
# =====================================================================

DATASET="nqopen"
MODEL_ARG="8b"

BOX_TYPES=(black-box white-box)
CONDITIONS=(c1 c2 c3 c4)
BOX_IDX=$((SLURM_ARRAY_TASK_ID / 4))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))
BOX="${BOX_TYPES[$BOX_IDX]}"
COND="${CONDITIONS[$COND_IDX]}"

# =====================================================================
# Cluster environment
# =====================================================================
module load compiler/gnu
module load devel/cuda

# --- Locate repo root robustly (fixes the earlier hard-coded-path bug) ---
# .env lives at the repo root and defines PROJECT_ROOT + HF_TOKEN.
# It is found via SLURM_SUBMIT_DIR (the dir you ran `sbatch` from).
ENV_FILE="${SLURM_SUBMIT_DIR:-$PWD}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "WARNING: .env not found at $ENV_FILE"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "$PROJECT_ROOT" || { echo "ERROR: cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
source venv/bin/activate

# --- Pre-flight: gated models need HF_TOKEN ---
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty. Set it in $ENV_FILE (export HF_TOKEN=...)."
    exit 1
fi
export HF_TOKEN

mkdir -p logs

PIPELINE="${BOX}/${DATASET}/${COND}_${DATASET}_pipeline.py"
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Pipeline not found -> $PIPELINE (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: ${PIPELINE}"
echo "  Dataset      : ${DATASET}"
echo "  Model        : ${MODEL_ARG}"
echo "  Box          : ${BOX}"
echo "  Condition    : ${COND}"
echo "  Array Task   : ${SLURM_ARRAY_TASK_ID}  (Job ${SLURM_ARRAY_JOB_ID})"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python "$PIPELINE" --model "$MODEL_ARG"