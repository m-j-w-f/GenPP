#!/bin/bash
# Launch a training job on a GPU node with local NVME data caching.
#
# This script:
#   1. Creates a unique job directory on the local NVME storage (/raid/$USER/<job_id>)
#   2. Copies the ICON data from the shared SSD (/shared/data/$USER/icon) to local NVME
#   3. Sets the GENPP_DATA_DIR environment variable to point to the local data
#   4. Runs the specified command
#   5. Cleans up the job directory after the job completes (success or failure)
#
# Usage:
#   ./launch_gpu_job.sh <command>
#
# Examples:
#   ./launch_gpu_job.sh pixi run python src/genpp/train --config-name base_drm data=icon data.batch_size=8
#   ./launch_gpu_job.sh pixi run wandb agent feik/genpp/wgvukbrf
#
# Notes:
#   - Data is copied per-job to ensure isolation when multiple jobs run on the same node
#   - Cleanup happens automatically, even if the job is interrupted (via trap)
#   - The script uses rsync for efficient copying

set -euo pipefail

# Check if a command was provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command>"
    echo "Example: $0 pixi run python src/genpp/train --config-name base_drm data=icon data.batch_size=8"
    echo "Example: $0 pixi run wandb agent feik/genpp/wgvukbrf"
    exit 1
fi

# Configuration
SOURCE_DATA_DIR="/shared/data/$USER/icon"
RAID_BASE_DIR="/raid"
USER_RAID_DIR="${RAID_BASE_DIR}/${USER}"

# Generate unique job ID using timestamp, PID, and random bytes for uniqueness
# Using 8 bytes of random data for better collision resistance in high-concurrency scenarios
JOB_ID="job_$(date +%Y%m%d_%H%M%S)_$$_$(head -c 8 /dev/urandom | xxd -p)"
JOB_DATA_DIR="${USER_RAID_DIR}/${JOB_ID}"

echo "=============================================="
echo "GenPP GPU Job Launcher"
echo "=============================================="
echo "Job ID: ${JOB_ID}"
echo "Source data: ${SOURCE_DATA_DIR}"
echo "Local data: ${JOB_DATA_DIR}"
echo "Command: $@"
echo "=============================================="

# Cleanup function to remove job data directory
cleanup() {
    local exit_code=$?
    echo ""
    echo "=============================================="
    echo "Cleaning up job data directory..."
    if [ -d "${JOB_DATA_DIR}" ]; then
        rm -rf "${JOB_DATA_DIR}"
        echo "Removed: ${JOB_DATA_DIR}"
    else
        echo "Job directory already cleaned up or doesn't exist"
    fi
    echo "=============================================="
    exit $exit_code
}

# Set trap to ensure cleanup runs on exit (success, failure, or interrupt)
trap cleanup EXIT INT TERM

# Verify source data exists
if [ ! -d "${SOURCE_DATA_DIR}" ]; then
    echo "ERROR: Source data directory does not exist: ${SOURCE_DATA_DIR}"
    echo "Please ensure the ICON data is available at the expected location."
    exit 1
fi

# Create user directory on /raid if it doesn't exist
if [ ! -d "${USER_RAID_DIR}" ]; then
    echo "Creating user directory on /raid: ${USER_RAID_DIR}"
    mkdir -p "${USER_RAID_DIR}"
fi

# Create job-specific data directory
echo "Creating job data directory: ${JOB_DATA_DIR}"
mkdir -p "${JOB_DATA_DIR}"

# Copy data to local NVME storage
echo "Copying data to local NVME storage..."
echo "This may take a few minutes depending on data size..."
START_TIME=$(date +%s)

# Use --no-group and --no-owner to avoid unnecessary overhead when copying to local storage
rsync -a --no-group --no-owner --info=progress2 "${SOURCE_DATA_DIR}/" "${JOB_DATA_DIR}/"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Data copy completed in ${DURATION} seconds"

# Export environment variable for the data directory
export GENPP_DATA_DIR="${JOB_DATA_DIR}"
echo "Set GENPP_DATA_DIR=${GENPP_DATA_DIR}"

echo ""
echo "=============================================="
echo "Starting job..."
echo "=============================================="

# Run the provided command and capture exit status for informative error messages
set +e  # Disable exit on error temporarily to capture the exit code
"$@"
CMD_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $CMD_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "ERROR: Command failed with exit code ${CMD_EXIT_CODE}"
    echo "Command: $@"
    echo "=============================================="
    exit $CMD_EXIT_CODE
fi

echo ""
echo "=============================================="
echo "Job completed successfully"
echo "=============================================="
