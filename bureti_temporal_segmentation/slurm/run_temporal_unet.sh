#!/bin/bash
#SBATCH --job-name=tea_mapping
#SBATCH --output=logs/tunet_%j.out
#SBATCH --error=logs/tunet_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=merit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:RTX5000:1
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=============================================="
echo "Temporal UNet: Experiments (E1, E3, E4)"
echo "=============================================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Start    : $(date)"
echo "=============================================="

# ====================
# Load .env
# ====================

ENV_FILE="${HOME}/bureti_temporal_segmentation/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env file not found at ${ENV_FILE}" >&2
    exit 1
fi
set -a
source "${ENV_FILE}"
set +a

echo "Loaded .env from ${ENV_FILE}"
echo "PROJECT_DIR : ${PROJECT_DIR}"
echo "DATA_DIR    : ${DATA_DIR}"
echo "OUTPUT_DIR  : ${OUTPUT_DIR}"

# ====================
# Training constants
# ====================

EPOCHS=65
BATCH_SIZE=8
LEARNING_RATE=0.0001

# Best single timestep from cross-timestep validation comparison
# 0=2023_growing, 1=2023_picking, 2=2024_growing, 3=2024_picking
BEST_TIMESTEP=2

# Same-year pair (2024 growing + 2024 picking)
ABLATION_TIMESTEPS="2,3"

# =========================
# Environment setup
# =========================

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '${CONDA_ENV_NAME}'" >&2
    exit 1
fi

NVIDIA_LIB="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="\
${NVIDIA_LIB}/nvjitlink/lib:\
${NVIDIA_LIB}/cudnn/lib:\
${NVIDIA_LIB}/cublas/lib:\
${NVIDIA_LIB}/cuda_runtime/lib:\
${NVIDIA_LIB}/cuda_nvrtc/lib:\
${NVIDIA_LIB}/cufft/lib:\
${NVIDIA_LIB}/curand/lib:\
${NVIDIA_LIB}/cusolver/lib:\
${NVIDIA_LIB}/cusparse/lib:\
/usr/lib/x86_64-linux-gnu:\
${LD_LIBRARY_PATH}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Env      : ${CONDA_ENV_NAME}"
echo "Environment ready"
echo ""

# ======================
# Helper function
# ======================

run_experiment() {
    local NAME="$1"
    local SCRIPT="$2"
    shift 2

    echo "##########################################################"
    echo "# EXPERIMENT: ${NAME}"
    echo "##########################################################"
    echo "Start: $(date)"
    echo ""

    python "${SCRIPT}" "$@"
    local EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "${NAME} completed successfully"
    else
        echo "ERROR: ${NAME} FAILED (exit code: ${EXIT_CODE})" >&2
    fi

    echo "End: $(date)"
    echo ""
    return ${EXIT_CODE}
}

# ====================================================
# E1 — Temporal UNet, 4 timesteps, 100% data
# ====================================================
for SEED in 42 123 456; do
    run_experiment \
        "E1 | Temporal UNet | 4T | 100% data | seed=${SEED}" \
        "temporal_unet_pipeline.py" \
        --data_dir        "${DATA_DIR}" \
        --output_dir      "${OUTPUT_DIR}" \
        --ssl4eo_weights  "${SSL4EO_WEIGHTS}" \
        --epochs          ${EPOCHS} \
        --batch_size      ${BATCH_SIZE} \
        --learning_rate   ${LEARNING_RATE} \
        --reference_timestep 2 \
        --timesteps       "0,1,2,3" \
        --train_fraction  1.0 \
        --seed            ${SEED}
done

# ====================================================
# E3 — Temporal UNet, 4 timesteps, 50% data
# ====================================================
for SEED in 42 123 456; do
    run_experiment \
        "E3 | Temporal UNet | 4T | 50% data | seed=${SEED}" \
        "temporal_unet_pipeline.py" \
        --data_dir        "${DATA_DIR}" \
        --output_dir      "${OUTPUT_DIR}" \
        --ssl4eo_weights  "${SSL4EO_WEIGHTS}" \
        --epochs          ${EPOCHS} \
        --batch_size      ${BATCH_SIZE} \
        --learning_rate   ${LEARNING_RATE} \
        --reference_timestep 2 \
        --timesteps       "0,1,2,3" \
        --train_fraction  0.5 \
        --seed            ${SEED}
done

# ====================================================
# E4 — Temporal UNet, 2 timesteps (2024 only), 100% data
# ====================================================
for SEED in 42 123 456; do
    run_experiment \
        "E4 | Temporal UNet | 2T | 100% data | seed=${SEED}" \
        "temporal_unet_pipeline.py" \
        --data_dir        "${DATA_DIR}" \
        --output_dir      "${OUTPUT_DIR}" \
        --ssl4eo_weights  "${SSL4EO_WEIGHTS}" \
        --epochs          ${EPOCHS} \
        --batch_size      ${BATCH_SIZE} \
        --learning_rate   ${LEARNING_RATE} \
        --reference_timestep 2 \
        --timesteps       ${ABLATION_TIMESTEPS} \
        --train_fraction  1.0 \
        --seed            ${SEED}
done


# Summary
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End Time: $(date)"
echo ""

conda deactivate 2>/dev/null || true