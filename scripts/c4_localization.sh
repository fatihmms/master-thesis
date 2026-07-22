#!/bin/bash
#SBATCH --job-name=fatih_kle_c4loc
#SBATCH --output=logs/c4loc_%A_%a.out
#SBATCH --error=logs/c4loc_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=02:00:00                          # ~15-25 judge calls per question, ONE file per task
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1                             # Llama-3.1-8B bf16 judge only
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =====================================================================
# C4 MoreHopQA error localization: WHICH STEP does hallucination start at?
#
# 4-task array, one result file per task (parallel-safe: each file has its
# own loc checkpoint and output, there is no shared report):
#   0 = 8b BB    1 = 8b WB    2 = mistral BB    3 = mistral WB
#
# For each file, an LLM judge checks every gold hop (from the frozen
# meta's question_decomposition) against every chain segment to find the
# FIRST factually wrong hop and the model step it falls on. Output:
#   <base>_loc8b_results.json       summary (first_error_hop/step
#                                   distributions, entropy-argmax agreement
#                                   vs chance) + details
#   <base>_loc8b_checkpoint.jsonl   resumable per-question checkpoint
#
# Reads --results (not --ckpt) so labels are the fixed-judge ones written
# by rejudge.py.
#
# PREREQUISITE: rejudge.py must have finished and been verified. Running
# this earlier localizes the Mistral files against stale self-judge labels
# (localization itself is label-independent; only the summary split is
# affected, and a re-run after rejudge refreshes labels from the input).
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/c4_localization.sh
# =====================================================================

module load compiler/gnu
module load devel/cuda

ENV_FILE="${SLURM_SUBMIT_DIR:-$PWD}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "WARNING: .env not found at $ENV_FILE"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "$PROJECT_ROOT" || { echo "ERROR: cannot cd to PROJECT_ROOT=$PROJECT_ROOT"; exit 1; }
source venv/bin/activate

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty. Set it in $ENV_FILE (export HF_TOKEN=...)."
    exit 1
fi
export HF_TOKEN

mkdir -p logs

# Same fixed judge as rejudge.py: hop labels and answer labels from ONE model.
LOC_JUDGE=8b

FILES=(
    results/black-box/morehopqa/c4_morehopqa_8b_BB_results.json
    results/white-box/morehopqa/c4_morehopqa_8b_WB_results.json
    results/black-box/morehopqa/c4_morehopqa_mistral_BB_results.json
    results/white-box/morehopqa/c4_morehopqa_mistral_WB_results.json
)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
FILE="${FILES[$TASK_ID]}"

if [[ ! -f "$FILE" ]]; then
    echo "ERROR: missing input file: $FILE (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: C4 MoreHopQA error localization"
echo "  Task      : ${TASK_ID}"
echo "  File      : ${FILE}"
echo "  Loc judge : ${LOC_JUDGE}"
echo "  Job       : ${SLURM_JOB_ID}"
echo "  Partition : ${SLURM_JOB_PARTITION}"
echo "  Node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python analysis/c4_error_localization.py --model "$LOC_JUDGE" --results "$FILE"

echo "=================================================="
echo "[$(date)] FINISHED task ${TASK_ID}: ${FILE}"
echo "=================================================="