#!/bin/bash
# Run a W&B agent in a detached GNU screen session on a specified GPU.
#
# Usage (direct):
#   ./src/genpp/scripts/run_wandb_agent.sh "<wandb command>" <device_number>
# Example (direct):
#   ./src/genpp/scripts/run_wandb_agent.sh 'wandb agent feik/genpp/9hmb7o41' 0
#
# Usage (via Pixi): the `wandb-agent` task is defined in `pyproject.toml` under
# `tool.pixi.feature.gpu.tasks`. Run with a GPU environment (e.g. `dev-gpu`):
#   pixi -e dev-gpu run wandb-agent 'wandb agent feik/genpp/9hmb7o41' 0
#
# Notes:
#  - The script creates a screen session named "wandb-agent-gpu<DEVICE>-<TIMESTAMP>".
#  - Attach with 'screen -r <session>' or list sessions with 'screen -ls'.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <wandb_command> <device_number>"
    echo "Example: $0 'wandb agent feik/genpp/9hmb7o41' 0"
    exit 1
fi

WANDB_CMD=$1
DEVICE=$2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="wandb-agent-gpu${DEVICE}-${TIMESTAMP}"

screen -dmS "${SESSION_NAME}" bash -c "CUDA_VISIBLE_DEVICES=${DEVICE} ${WANDB_CMD}"
echo "Started '${WANDB_CMD}' on GPU ${DEVICE} in screen session '${SESSION_NAME}'"
echo "Use 'screen -r ${SESSION_NAME}' to attach to the session"
echo "Use 'screen -ls' to list all sessions"
