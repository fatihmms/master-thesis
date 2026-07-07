#!/bin/bash
#SBATCH --job-name=fatih_selfcheck_c2
#SBATCH --output=logs/selfcheck_c2_%A_%a.out     # %A = array job id, %a = task id (0-5)
#SBATCH --error=logs/selfcheck_c2_%A_%a.err
#SBATCH --time=00:30:00                          # 10 NLI pairs/question, batched -> 1000 q in minutes
#SBATCH --partition=gpu_a100_short                     # Override at submit: sbatch --partition=gpu_h100_short ... (smoke)
#SBATCH --gres=gpu:1                             # DeBERTa-v2-xlarge fp16 only; no generator loaded
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5                              # 0-2 -> 8b x {triviaqa,nqopen,morehopqa}, 3-5 -> mistral x same

# =====================================================================
# SelfCheckGPT-NLI baseline on the C2 (holistic CoT) setting
# [post-hoc, black-box, response-level] — scores the saved C2 BB responses
# with DeBERTa-v2-xlarge-MNLI; no generation.
# Tasks whose C2 BB source file does not exist yet (e.g. mistral
# triviaqa/nqopen still queued) fail fast with a clear FileNotFoundError
# and can be resubmitted later:
#   sbatch --array=3,4 scripts/selfcheck_c2.sh
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/selfcheck_c2.sh
# Smoke test (N=50, short partition):
#   sbatch --partition=gpu_h100_short --time=00:20:00 --export=ALL,N_LIMIT=50 --array=2 scripts/selfcheck_c2.sh
# =====================================================================

MODELS=(8b mistral)
DATASETS=(triviaqa nqopen morehopqa)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 3))
MODEL_ARG="${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DS_IDX]}"

N_LIMIT="${N_LIMIT:-0}"              # 0 = all questions; override for smoke

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

# HF_TOKEN not strictly required (DeBERTa is not gated) — warn, don't abort.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is empty (ok for DeBERTa, set it in $ENV_FILE if downloads fail)."
fi
export HF_TOKEN

mkdir -p logs

PIPELINE="baselines/selfcheck_nli_c2_pipeline.py"
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Pipeline not found -> $PIPELINE (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: ${PIPELINE}"
echo "  Dataset      : ${DATASET}"
echo "  Model (src)  : ${MODEL_ARG}"
echo "  N limit      : ${N_LIMIT}"
echo "  Array Task   : ${SLURM_ARRAY_TASK_ID}  (Job ${SLURM_ARRAY_JOB_ID})"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python "$PIPELINE" --model "$MODEL_ARG" --dataset "$DATASET" --n "$N_LIMIT"
