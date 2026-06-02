#!/usr/bin/env bash
# Start the ARX-A5 X-VLA OpenPI-compatible WebSocket policy server.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XVLA_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
XVLA_ENV="${XVLA_ENV:-XVLA}"
XVLA_PORT="${XVLA_PORT:-8010}"
XVLA_HOST="${XVLA_HOST:-0.0.0.0}"
XVLA_DEVICE="${XVLA_DEVICE:-cuda}"
XVLA_STEPS="${XVLA_STEPS:-10}"
CHECKPOINT_DIR="${XVLA_ROOT}/checkpoints/xvla_finetune_single_task_ee6d/put_shrimp_in_pot-20260602_181531/ckpt-15000"
OUTPUT_DIR="${OUTPUT_DIR:-${XVLA_ROOT}/logs/arx_a5-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "[xvla] ERROR: checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    exit 2
fi

EXTRA_ARGS=()
if [[ -n "${PROCESSOR_DIR:-}" ]]; then
    EXTRA_ARGS+=(--processor_path "${PROCESSOR_DIR}")
fi
if [[ -n "${LORA_DIR:-}" ]]; then
    EXTRA_ARGS+=(--LoRA_path "${LORA_DIR}")
fi

echo "[xvla] root=${XVLA_ROOT}"
echo "[xvla] checkpoint=${CHECKPOINT_DIR}"
echo "[xvla] bind=ws://${XVLA_HOST}:${XVLA_PORT}"
echo "[xvla] output=${OUTPUT_DIR}"
echo "[xvla] conda_env=${XVLA_ENV}"

cd "${XVLA_ROOT}"
exec conda run --no-capture-output -n "${XVLA_ENV}" python deploy.py \
    --model_path "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --host "${XVLA_HOST}" \
    --port "${XVLA_PORT}" \
    --device "${XVLA_DEVICE}" \
    --steps "${XVLA_STEPS}" \
    --disable_slurm \
    "${EXTRA_ARGS[@]}"
