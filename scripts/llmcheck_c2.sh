#!/bin/bash
#SBATCH --job-name=fatih_llmcheck_c2
#SBATCH --output=logs/llmcheck_c2_%A_%a.out      # %A = array job id, %a = task id (0-5)
#SBATCH --error=logs/llmcheck_c2_%A_%a.err
#SBATCH --time=06:00:00                          # ~2-5 s/question (1 fwd pass + CPU SVD) -> 1000 q well under 6h
#SBATCH --partition=gpu_h100                     # Override at submit: sbatch --partition=gpu_h100_short ... (smoke)
#SBATCH --gres=gpu:1                             # single teacher-forced fwd pass; 8B bf16 + eager attn fits on 1 GPU
#SBATCH --cpus-per-task=8                        # SVD / eigen scores run on CPU (official code path)
#SBATCH --mem=32G
#SBATCH --array=0-5                              # 0-2 -> 8b x {triviaqa,nqopen,morehopqa}, 3-5 -> mistral x same

# =====================================================================
# LLM-Check baseline on the C2 (holistic CoT) setting  [post-hoc, white-box]
# Scores the saved judge_candidate chains of the C2 WB runs; no generation.
# 6-task array = 2 models x 3 datasets (mirrors the 8-task pattern of the
# condition scripts). Tasks whose C2 WB source file does not exist yet
# (e.g. mistral triviaqa/nqopen still queued) fail fast with a clear
# FileNotFoundError and can be resubmitted later:
#   sbatch --array=4,5 scripts/llmcheck_c2.sh
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/llmcheck_c2.sh
# Smoke test (N=50, short partition):
#   sbatch --partition=gpu_h100_short --time=00:30:00 --export=ALL,N_LIMIT=50 scripts/llmcheck_c2.sh
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

# --- Pre-flight: gated models need HF_TOKEN ---
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty. Set it in $ENV_FILE (export HF_TOKEN=...)."
    exit 1
fi
export HF_TOKEN

# --- Pre-flight: official LLM-Check repo must be cloned at the repo root ---
if [[ ! -d "LLM_Check_Hallucination_Detection" ]]; then
    echo "ERROR: LLM_Check_Hallucination_Detection/ not found at repo root."
    echo "  git clone https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection.git"
    exit 1
fi

mkdir -p logs

PIPELINE="baselines/llmcheck_c2_pipeline.py"
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Pipeline not found -> $PIPELINE (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: ${PIPELINE}"
echo "  Dataset      : ${DATASET}"
echo "  Model        : ${MODEL_ARG}"
echo "  N limit      : ${N_LIMIT}"
echo "  Array Task   : ${SLURM_ARRAY_TASK_ID}  (Job ${SLURM_ARRAY_JOB_ID})"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python "$PIPELINE" --model "$MODEL_ARG" --dataset "$DATASET" --n "$N_LIMIT"
