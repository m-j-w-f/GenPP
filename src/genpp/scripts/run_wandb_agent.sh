#!/bin/bash
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
