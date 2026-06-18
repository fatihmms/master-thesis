#!/bin/bash
#SBATCH --job-name=kle_nqopen_bb
#SBATCH --output=kle_nqopen_bb_%A_%a.out   # %A = array job id, %a = task index (separate log per condition)
#SBATCH --error=kle_nqopen_bb_%A_%a.err
#SBATCH --time=12:00:00                     # each task runs both models back-to-back; raise for full 1000-q runs
#SBATCH --partition=gpu_h100                # GPU queue
#SBATCH --gres=gpu:1                        # 1 GPU per array task
#SBATCH --cpus-per-task=8                   # 8 CPUs per task
#SBATCH --mem=32G                           # 32 GB RAM per task
#SBATCH --array=0-2                         # 3 black-box conditions: 0->C1, 1->C2, 2->C3

# ---- fixed for this script ----
ACCESS="black-box"
DATASET="nqopen"

# ---- condition (from array index) + models (looped inside the task) ----
SCRIPTS=(c1 c2 c3 c4)            # black-box conditions
MODELS=(8b mistral)           # model aliases passed via --model
NAME=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}
PIPELINE="${ACCESS}/${DATASET}/${NAME}_${DATASET}_pipeline.py"

# 1. Load the same modules you loaded on the login node
module load compiler/gnu
module load devel/cuda

# 2. Go to the repo root (root_dir resolution + ./results/ depend on this)
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

# 5. Run the pipeline for each model, sequentially (one failure does not stop the other).
for MODEL in "${MODELS[@]}"; do
    echo "--------------------------------------------------"
    echo "[$(date)] RUN   ${PIPELINE}  --model ${MODEL}"
    python "$PIPELINE" --model "$MODEL"
    echo "[$(date)] DONE  ${PIPELINE}  --model ${MODEL}  (exit $?)"
done

echo "=================================================="
echo "[$(date)] TASK COMPLETE  ${PIPELINE}"