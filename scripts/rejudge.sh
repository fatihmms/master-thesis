#!/bin/bash
#SBATCH --job-name=fatih_kle_rejudge
#SBATCH --output=logs/rejudge_%j.out
#SBATCH --error=logs/rejudge_%j.err
#SBATCH --time=04:00:00                          # ~24k greedy judge calls (8 tokens each); typically well under 2h
#SBATCH --partition=gpu_h100                     # short job: sbatch --partition=gpu_h100_short scripts/rejudge.sh also works
#SBATCH --gres=gpu:1                             # Llama-3.1-8B bf16 only, no generator and no NLI model
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =====================================================================
# Fixed-judge re-labeling of ALL saved results  (IN-PLACE)
#
# Re-labels every c1-c4 x {black,white}-box x {triviaqa,nqopen,morehopqa}
# result file with a single fixed judge (Llama-3.1-8B-Instruct).
#
#   Mistral files (24) -> every record re-judged
#   Llama-8B files (24) -> labels identical by construction; config stamping
#                          only, plus a 50-record verification
#
# The result files stay indistinguishable from native fixed-judge pipeline
# output. All provenance of the re-labeling (agreement, kappa, old/new AUROC,
# disagreements, original labels) lands in results/rejudge_report.json.
#
# Baselines are NOT touched: they read judge_correct straight out of the C2
# result files, so re-running them AFTER this job picks up the new labels.
#
# Submit FROM the repo root:  cd <repo> && sbatch scripts/rejudge.sh
# Dry run first (login node, no GPU needed):
#   python analysis/rejudge.py --dry-run results/*-box/*/c[1-4]_*_results.json
# =====================================================================

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

# --- Pre-flight: the judge is a gated model ---
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty. Set it in $ENV_FILE (export HF_TOKEN=...)."
    exit 1
fi
export HF_TOKEN

mkdir -p logs

FILES=(results/black-box/*/c[1-4]_*_results.json results/white-box/*/c[1-4]_*_results.json)
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "ERROR: no result files matched (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] STARTED: fixed-judge re-labeling"
echo "  Judge        : meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  Files        : ${#FILES[@]}"
echo "  Job          : ${SLURM_JOB_ID}"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Node         : $(hostname)"
echo "  Repo root    : ${PROJECT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

python analysis/rejudge.py --verify 50 "${FILES[@]}"

echo "=================================================="
echo "[$(date)] FINISHED"
echo "=================================================="