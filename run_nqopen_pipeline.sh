#!/bin/bash
#SBATCH --job-name=kle_nqopen
#SBATCH --output=kle_nqopen_%A_%a.out    # %A = array job id, %a = task index (separate log per condition)
#SBATCH --error=kle_nqopen_%A_%a.err
#SBATCH --time=12:00:00                   # NOTE: each task now runs BOTH models back-to-back,
                                          # so wall-time ~doubles. For full 1000-question C4
                                          # runs you likely need to raise this (e.g. 24:00:00).
#SBATCH --partition=gpu_h100              # GPU queue
#SBATCH --gres=gpu:1                      # 1 GPU per array task
#SBATCH --cpus-per-task=8                 # 8 CPUs per task
#SBATCH --mem=32G                         # 32 GB RAM per task (both 8B and 7B fit fine)
#SBATCH --array=0-3                       # 4 tasks = one per condition (0->C1, 1->C2, 2->C3, 3->C4)

# ---- dataset (one script per dataset, so just change this name to reuse) ----
SCRIPTS=(c1 c2 c3 c4)
MODELS=(8b mistral)
NAME=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}

# c1-c3 live under black-box/, c4 under white-box/
if [[ "$NAME" == "c4" ]]; then
    ACCESS="white-box"
else
    ACCESS="black-box"
fi
PIPELINE="${ACCESS}/${DATASET}/${NAME}_${DATASET}_pipeline.py"

# 1. Load the same modules you loaded on the login node
module load compiler/gnu
module load devel/cuda

# 2. Go to the working directory (c1-c4 + frozen_utils.py must live here)
cd /home/ka/ka_ksri/ka_cx3082/master-thesis

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Hugging Face token (required for gated Llama / Mistral models)
export HF_TOKEN="token"

# --- pre-flight ---
if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: $PIPELINE cannot be found (CWD: $(pwd))"
    exit 1
fi

echo "=================================================="
echo "[$(date)] START TASK  ${PIPELINE}"
echo "  array task : ${SLURM_ARRAY_TASK_ID}  (job ${SLURM_ARRAY_JOB_ID})"
echo "  node       : $(hostname)"
echo "  models     : ${MODELS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=================================================="

# 5. Run the pipeline for each model, sequentially.
#    A failure on one model does NOT stop the other (no `set -e`).
for MODEL in "${MODELS[@]}"; do
    echo "--------------------------------------------------"
    echo "[$(date)] RUN   ${PIPELINE}  --model ${MODEL}"
    python "$PIPELINE" --model "$MODEL"
    echo "[$(date)] DONE  ${PIPELINE}  --model ${MODEL}  (exit $?)"
done

echo "=================================================="
echo "[$(date)] TASK COMPLETE  ${PIPELINE}"