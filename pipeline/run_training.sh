#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ---------------------------------------------------
# change training dataset in meta.json
TASK_NAME="put_shrimp_in_pot"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/xvla_finetune_single_task_ee6d/${TASK_NAME}-$(date +%Y%m%d)_$(date +%H%M%S)"
BATCH_SIZE=32       # single GPU
LEARNING_RATE=5e-5
SAVE_INTERVAL=5000
# ---------------------------------------------------

MODELS="${MODELS:-${REPO_ROOT}/checkpoints/X-VLA-Pt}"
TRAIN_METAS_PATH="${TRAIN_METAS_PATH:-${SCRIPT_DIR}/meta.json}"
ACTION_MODE="${ACTION_MODE:-arx_ee6d}"
LEARNING_COEF="${LEARNING_COEF:-0.1}"
ITERS="${ITERS:-50000}"
FREEZE_STEPS="${FREEZE_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
NUM_PROCESSES="${NUM_PROCESSES:-$(python -c 'import os; print(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))')}"
NUM_MACHINES="${NUM_MACHINES:-1}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"
# Passing port 0 through accelerate 1.2.1 can leave torch connecting to port 0.
# Resolve it here instead. Set MAIN_PROCESS_PORT to a fixed non-zero value to override.
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
if [[ "${MAIN_PROCESS_PORT}" == "0" ]]; then
    MAIN_PROCESS_PORT="$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi

accelerate launch \
    --multi_gpu \
    --mixed_precision "${MIXED_PRECISION:-bf16}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --dynamo_backend "${DYNAMO_BACKEND}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    train.py \
    --models "${MODELS}" \
    --train_metas_path "${TRAIN_METAS_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --action_mode "${ACTION_MODE}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --learning_coef "${LEARNING_COEF}" \
    --iters "${ITERS}" \
    --freeze_steps "${FREEZE_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}" \
    "$@"
