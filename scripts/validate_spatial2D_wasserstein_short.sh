#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/spatial2D_common.sh"

cd "${PROJECT_ROOT}"

SPATIAL2D_STORAGE_ROOT="${SPATIAL2D_STORAGE_ROOT:-$(default_spatial2d_storage_root)}"
TRAIN_RUN_BASE="${TRAIN_RUN_BASE:-${SPATIAL2D_STORAGE_ROOT}/run/train/spatial2D}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
if [[ -z "${TRAIN_RUN_DIR}" ]]; then
  if completed_train_run="$(latest_completed_train_run "${TRAIN_RUN_BASE}")"; then
    TRAIN_RUN_DIR="${completed_train_run}"
  else
    die "TRAIN_RUN_DIR is empty and no completed spatial2D training run was found under ${TRAIN_RUN_BASE}."
  fi
fi

CHECKPOINT_SUBSTR="${CHECKPOINT_SUBSTR:-last}"
POOLING_METHOD="${POOLING_METHOD:-no_cls}"
PRIOR_LOW="${PRIOR_LOW:-0,0}"
PRIOR_HIGH="${PRIOR_HIGH:-1,1}"
SEED="${SEED:-12345}"
DEVICE="${DEVICE:-cuda}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${TRAIN_RUN_DIR}/wasserstein_short_validation/$(date +%Y-%m-%d_%H-%M-%S)}"

activate_env
assert_python_stack
assert_spatial2d_data
ensure_spatial2d_extension

mkdir -p "${VALIDATION_OUTPUT_DIR}"

log "Running short latent metric probe."
python scripts/probe_spatial2D_latent_distance.py \
  --run-dir "${TRAIN_RUN_DIR}" \
  --checkpoint-substr "${CHECKPOINT_SUBSTR}" \
  --output-dir "${VALIDATION_OUTPUT_DIR}/probe" \
  --n-theta "${N_THETA:-4}" \
  --pooling-method "${POOLING_METHOD}" \
  --metric pairwise_wasserstein \
  --prior-low "${PRIOR_LOW}" \
  --prior-high "${PRIOR_HIGH}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --short

log "Running short Wasserstein inference with the original qt stopping logic."
python src/inference.py \
  inference=spatial2D \
  debug=spatial2D \
  run_folder_path="${TRAIN_RUN_DIR}" \
  folder_name="${VALIDATION_OUTPUT_DIR}/inference_wasserstein" \
  checkpoint_substr="${CHECKPOINT_SUBSTR}" \
  system.pooling_method="${POOLING_METHOD}" \
  system.metric=pairwise_wasserstein \
  abc.num_particles="${ABC_NUM_PARTICLES:-20}" \
  abc.k="${ABC_K:-2}" \
  abc.q_threshold="${ABC_Q_THRESHOLD:-0.90}" \
  abc.max_generations="${ABC_MAX_GENERATIONS:-6}" \
  abc.num_workers="${ABC_NUM_WORKERS:-4}" \
  abc.simulation_batch_size="${ABC_SIMULATION_BATCH_SIZE:-4}" \
  abc.max_pending_simulations="${ABC_MAX_PENDING_SIMULATIONS:-8}"

log "Short validation outputs:"
printf '  probe report: %s\n' "${VALIDATION_OUTPUT_DIR}/probe/validation_report.json"
printf '  inference generations: %s\n' "${VALIDATION_OUTPUT_DIR}/inference_wasserstein/abc_generations.npy"
