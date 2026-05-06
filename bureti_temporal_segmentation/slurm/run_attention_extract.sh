#!/bin/bash
#SBATCH --job-name=attn_extract
#SBATCH --output=logs/attn_extract_%j.out
#SBATCH --error=logs/attn_extract_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=merit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:RTX5000:1

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

# E1 — seed
python extract_attention_weights.py \
    --checkpoint_path "${OUTPUT_DIR}/run_dir/models/temporal_unet_best.keras" \
    --ssl4eo_weights "${SSL4EO_WEIGHTS}" \
    --data_dir "${DATA_DIR}" \
    --norm_params "${OUTPUT_DIR}/run_dir/models/norm_params.pkl" \
    --output_dir "${OUTPUT_DIR}/attention_analysis/E1_seed123" \
    --timesteps "0,1,2,3" \
    --reference_timestep 2

# E4 — seed
python extract_attention_weights.py \
    --checkpoint_path "${OUTPUT_DIR}/run_dir/models/temporal_unet_best.keras" \
    --ssl4eo_weights "${SSL4EO_WEIGHTS}" \
    --data_dir "${DATA_DIR}" \
    --norm_params "${OUTPUT_DIR}/run_dir/models/norm_params.pkl" \
    --output_dir "${OUTPUT_DIR}/attention_analysis/E4_seed42" \
    --timesteps ${ABLATION_TIMESTEPS} \
    --reference_timestep 2

conda deactivate 2>/dev/null || true