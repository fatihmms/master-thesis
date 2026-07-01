#!/bin/bash
#SBATCH --job-name=fatihm_kle_nqopen_mistral_full
#SBATCH --output=logs/nqopen_mistral_%A_%a.out       # Redirect logs to logs/ directory to keep the root clean
#SBATCH --error=logs/nqopen_mistral_%A_%a.err
#SBATCH --time=12:00:00                     # ~10 hours execution time per pipeline + 5 hours safety margin
#SBATCH --partition=gpu_h100             # BwUniCluster 3.0 GPU A100 queue
#SBATCH --gres=gpu:1                        # Each array task requests exactly 1 dedicated A100 GPU
#SBATCH --cpus-per-task=8                   # 8 CPU cores per task (Proven working specification)
#SBATCH --mem=32G                           # 32 GB RAM per task (Proven working specification)
#SBATCH --array=0-7                         # Total 8 parallel jobs: 0-3 Black-Box (c1-c4), 4-7 White-Box (c1-c4)

# =====================================================================
# 2D Array Mapping Logic
# =====================================================================
# Array ID: 0,1,2,3 -> Black-Box (c1, c2, c3, c4)
# Array ID: 4,5,6,7 -> White-Box (c1, c2, c3, c4)

BOX_TYPES=(black-box white-box)
CONDITIONS=(c1 c2 c3 c4)

# Calculate matrix indices using integer division and modulo operations
BOX_IDX=$((SLURM_ARRAY_TASK_ID / 4))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))

BOX="${BOX_TYPES[$BOX_IDX]}"
COND="${CONDITIONS[$COND_IDX]}"

# Dynamic path configuration aligned with the repository's structure
PIPELINE="${BOX}/nqopen/${COND}_nqopen_pipeline.py"

# =====================================================================
# Cluster Environment Setup
# =====================================================================
module load compiler/gnu
module load devel/cuda

# Navigate to the project root directory
cd /home/ka/ka_ksri/ka_cx3082/master-thesis
source venv/bin/activate

# Hugging Face token environment variable for gated LLM access
export HF_TOKEN="token"

# Pre-flight check to verify target script existence
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Pipeline script not found -> $PIPELINE (CWD: $(pwd))"
    exit 1
fi

# Ensure the logs directory is present
mkdir -p logs

echo "=================================================="
echo "[$(date)] STARTED: ${PIPELINE}"
echo "  Mode (Box)   : ${BOX}"
echo "  Condition    : ${COND}"
echo "  Array Task   : ${SLURM_ARRAY_TASK_ID} (Global Job ID: ${SLURM_ARRAY_JOB_ID})"
echo "  Node Name    : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

# =====================================================================
# Execute Pipeline (With argparse configuration for dynamic models)
# =====================================================================
python "$PIPELINE" --model mistral