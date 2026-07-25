#!/bin/bash
#SBATCH --job-name=fatih_kle_morehopqa_70b
#SBATCH --output=logs/morehopqa_70b_%A_%a.out    # %A = array job id, %a = task id (0-7)
#SBATCH --error=logs/morehopqa_70b_%A_%a.err
#SBATCH --time=01:00:00                          # C4 cells estimated 25-35h at N=1000; 72h is the partition max
#SBATCH --partition=gpu_h100_il                     # fallback with 4x H100 nodes: sbatch --partition=gpu_h100_il (48h max!)
#SBATCH --gres=gpu:4                             # 70B bf16 ~140GB weights -> sharded over 4x H100 via device_map=auto
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-7                              # 0-3 -> black-box c1-c4, 4-7 -> white-box c1-c4

# =====================================================================
# MoreHopQA  x  Meta-Llama-3.1-70B-Instruct   (70B-specific pipeline files)
#
# Runs the c*_morehopqa_70b_pipeline.py copies: generator hardcoded to
# 3.1-70B, self-judge BY DESIGN (do NOT rejudge these results), WB files
# use HIDDEN_LAYER=40 (proportional mid-depth for 80 layers).
#
# Optional first argument = number of questions (passed as --n):
#   sbatch scripts/morehopqa_70b.sh              # full N=1000
#   sbatch scripts/morehopqa_70b.sh 500          # halved run
#   sbatch --time=03:00:00 scripts/morehopqa_70b.sh 10   # smoke test
#     (smoke test on gpu_h100, NOT gpu_h100_short: 30-min limit there is
#      shorter than the ~10-15 min it takes just to load 70B weights)
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/morehopqa_70b.sh
# =====================================================================

DATASET="morehopqa"
N_ARG="${1:-1000}"

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

# --- Locate repo root robustly ---
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

# --- Pre-flight: 3.1-70B is gated (license must be accepted on HF) ---
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty. Set it in $ENV_FILE (export HF_TOKEN=...)."
    exit 1
fi
export HF_TOKEN

mkdir -p logs

PIPELINE="${BOX}/${DATASET}/${COND}_${DATASET}_70b_pipeline.py"
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Pipeline not found -> $PIPELINE (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: ${PIPELINE}"
echo "  Dataset      : ${DATASET}"
echo "  Model        : Meta-Llama-3.1-70B-Instruct (hardcoded)"
echo "  N questions  : ${N_ARG}"
echo "  Box          : ${BOX}"
echo "  Condition    : ${COND}"
echo "  Array Task   : ${SLURM_ARRAY_TASK_ID}  (Job ${SLURM_ARRAY_JOB_ID})"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python "$PIPELINE" --n "$N_ARG"

echo "=================================================="
echo "[$(date)] FINISHED: ${PIPELINE}"
echo "=================================================="
