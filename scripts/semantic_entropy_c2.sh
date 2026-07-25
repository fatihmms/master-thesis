#!/bin/bash
#SBATCH --job-name=se_c2_fatih
#SBATCH --output=logs/se_c2_%j.out
#SBATCH --error=logs/se_c2_%j.err
#SBATCH --time=00:10:00                          # pure numpy/sklearn re-score over 6 cells x ~1000 rows; no model, no GPU -> seconds of compute
#SBATCH --partition=gpu_h100_short   
#SBATCH --gres=gpu:1              # no CPU-only partition exists in this project's convention (see scripts/*.sh); requested here purely for a fast allocation slot, GPU not used
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# =====================================================================
# Semantic Entropy baseline (Kuhn 2023 / Farquhar 2024) on C2  [post-hoc]
# Recomputes se_discrete + se_weighted from the semantic_ids/log_lik_per_sem_id
# already stored in the saved C2 BB result files. No generation, no NLI
# model, no torch — pure numpy + sklearn over all 2 models x 3 datasets in
# one process (baselines/semantic_entropy_posthoc.py has no --model/--dataset
# args; it always does all 6 cells in a single run).
#
# Runs anywhere; this script exists only so submission matches the other two
# baseline jobs (scripts/llmcheck_c2.sh, scripts/selfcheck_c2.sh). Running it
# directly on a login node is equally valid:
#   cd <repo> && python baselines/semantic_entropy_posthoc.py
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/semantic_entropy_c2.sh
# =====================================================================

# =====================================================================
# Cluster environment
# =====================================================================
# --- Locate repo root robustly ---
# .env lives at the repo root and defines PROJECT_ROOT.
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

mkdir -p logs

PIPELINE="baselines/semantic_entropy_posthoc.py"
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Pipeline not found -> $PIPELINE (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: ${PIPELINE}"
echo "  Job          : ${SLURM_JOB_ID}"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
echo "=================================================="

python "$PIPELINE"
