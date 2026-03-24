#!/bin/bash -l

#PBS -N process_tensors                # Job name
#PBS -S /bin/bash                      # set the executing shell
#PBS -q rc_big
#PBS -l cpunum_job=1                   # use 1 CPU
#PBS -l memsz_job=4gb                  # total memory for job
#PBS -l vmemsz_job=64gb                # total virtual memory
#PBS -l elapstim_req=02:00:00          # max runtime: 10 minutes
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/scripts/logs/process_tensors.log
#PBS -j o                              # concatenate stderr and stdout

# --- Read task parameters from task list using sub-request number ---
if [ -z "$TASK_LIST" ] || [ -z "$PBS_SUBREQNO" ]; then
    echo "ERROR: TASK_LIST and PBS_SUBREQNO must be set."
    echo "This script is meant to be submitted as an array job."
    exit 1
fi

TASK_LINE=$(sed -n "${PBS_SUBREQNO}p" "$TASK_LIST")
if [ -z "$TASK_LINE" ]; then
    echo "ERROR: No task found for sub-request number ${PBS_SUBREQNO}"
    exit 1
fi

export JOB_TYPE=$(echo $TASK_LINE | awk '{print $1}')
export YEAR=$(echo $TASK_LINE | awk '{print $2}')
export MONTH=$(echo $TASK_LINE | awk '{print $3}')
export DAY=$(echo $TASK_LINE | awk '{print $4}')

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Sub-request: ${PBS_SUBREQNO}"
echo "Processing: JOB_TYPE=$JOB_TYPE, YEAR=$YEAR, MONTH=$MONTH, DAY=$DAY"
echo "=========================================="

# Store the script directory
SCRIPT_DIR=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/scripts

# Repository root
REPO_ROOT=/hpc/uhome/extmfeik/GenPP

# Change to repository root for pixi
cd "$REPO_ROOT" || exit 1

# Add pixi to PATH if not already present
if [ -f "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
elif [ -f "$HOME/.local/bin/pixi" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify pixi is available
if ! command -v pixi &> /dev/null; then
    echo "ERROR: pixi command not found in PATH"
    exit 1
fi

# Verify the script exists
if [ ! -f "$SCRIPT_DIR/process_tensors.py" ]; then
    echo "ERROR: process_tensors.py not found in $SCRIPT_DIR"
    exit 1
fi

# Activate pixi environment and run the Python script
eval "$(pixi shell-hook)"
pixi run python "$SCRIPT_DIR/process_tensors.py"

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
