#!/bin/bash
#SBATCH --job-name=tea_mapping
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=merit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:RTX5000:1
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=========================================="
echo "DeepLabV3+: Experiments (E2, E5)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="

# ===================
# Configuration
# ===================

PROJECT_DIR="${HOME}/bureti_temporal_segmentation"
DATA_DIR="${PROJECT_DIR}/data/Buret_Multitemporal_Data"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
SSL4EO_WEIGHTS="${PROJECT_DIR}/ssl4eo_resnet50_12ch.keras"

CONDA_ENV_NAME="tea_seg"

EPOCHS=65
BATCH_SIZE=8
LEARNING_RATE=0.0001

# Best single timestep from cross-timestep comparison
# 0=2023_growing, 1=2023_picking, 2=2024_growing, 3=2024_picking
BEST_TIMESTEP=2

# 2-timestep ablation: same-year pair (2024 growing + 2024 picking)
ABLATION_TIMESTEPS="2,3"

# ========================
# Experiment Setup
# ========================

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

NVIDIA_LIB="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIB}/nvjitlink/lib:${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cuda_nvrtc/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "Environment ready"

# ===========================
# Helper function
# ===========================

run_experiment() {
    local NAME="$1"
    local SCRIPT="$2"
    shift 2

    echo ""
    echo "##########################################################"
    echo "# EXPERIMENT: ${NAME}"
    echo "##########################################################"
    echo "Start: $(date)"
    echo ""

    python "${SCRIPT}" "$@"
    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "${NAME} completed successfully"
    else
        echo "${NAME} FAILED (exit code: ${EXIT_CODE})"
    fi

    echo "End: $(date)"
    echo ""
    return $EXIT_CODE
}

# =============================================================================
# Baseline: DeepLabV3+, T2 (2024_growing), 100% data
# Runs only the best timestep identified from cross-timestep comparison.
# =============================================================================
for SEED in 42 123 456; do
    run_experiment \
        "E2 | DeepLabV3+ | T2 | 100% data" \
        "deeplabv3_pipeline.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --ssl4eo_weights "${SSL4EO_WEIGHTS}" \
        --timestep ${BEST_TIMESTEP} \
        --train_fraction 1.0 \
        --seed ${SEED}
done
# =============================================================================
# Data Scarcity Ablation: DeepLabV3+, T2 (2024_growing), 50% data
# =============================================================================
for SEED in 42 123 456; do
    run_experiment \
        "E5 | DeepLabV3+ | T2 | 50% data" \
        "deeplabv3_pipeline.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --ssl4eo_weights "${SSL4EO_WEIGHTS}" \
        --timestep ${BEST_TIMESTEP} \
        --train_fraction 0.5 \
        --seed ${SEED}
done

# Summary
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "End Time: $(date)"
echo ""

conda deactivate 2>/dev/null || true