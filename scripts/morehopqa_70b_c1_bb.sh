#!/bin/bash
#SBATCH --job-name=70b_c1_bb_fatih_kle_morehopqa
#SBATCH --output=logs/morehopqa_70b_c1_bb_%j.out
#SBATCH --error=logs/morehopqa_70b_c1_bb_%j.err
#SBATCH --time=14:00:00                          # smoke: 32.9s/q x1000 = ~9.1h, x1.5 safety margin (72h partition max)
#SBATCH --partition=gpu_h100                     # fallback with 4x H100 nodes: sbatch --partition=gpu_h100_il (48h max, fits this cell)
#SBATCH --gres=gpu:4                             # 70B bf16 ~140GB weights -> sharded over 4x H100 via device_map=auto
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G                                # smoke test used ~5-7.5GB host RAM (not GPU VRAM); 64G is a generous margin

# =====================================================================
# MoreHopQA C1 black-box  x  Meta-Llama-3.1-70B-Instruct
#
# One of 8 cells submitted SEPARATELY (not as an array) so each gets its own
# --time tailored to its real cost instead of sharing one blanket value.
# Submitting separately does not reduce parallelism vs an array: both are 8
# independent 4-GPU resource requests to the same scheduler, which runs as
# many concurrently as capacity allows either way. What tailoring buys is
# backfill: a job requesting only what it needs is far easier for the
# scheduler to slot into a free gap than one requesting a blanket worst-case
# number for every cell.
#
# Runs the c1_morehopqa_70b_pipeline.py copy: generator hardcoded to
# 3.1-70B, self-judge BY DESIGN (do NOT rejudge this result).
#
# Optional first argument = number of questions (passed as --n):
#   sbatch morehopqa_70b_c1_bb.sh              # full N=1000
#   sbatch morehopqa_70b_c1_bb.sh 500          # halved run
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/morehopqa_70b_c1_bb.sh
# =====================================================================

DATASET="morehopqa"
N_ARG="${1:-1000}"
BOX="black-box"
COND="c1"

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
echo "  Job          : ${SLURM_JOB_ID}"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python "$PIPELINE" --n "$N_ARG"

echo "=================================================="
echo "[$(date)] FINISHED: ${PIPELINE}"
echo "=================================================="
