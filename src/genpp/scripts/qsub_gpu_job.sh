#!/bin/bash
# Submit this script with qsub to run a training job on a GPU node with local NVME data caching.
#
# This script is designed for the NQSV batch system and cannot accept command-line arguments.
# Instead, edit the COMMAND variable below before submitting with qsub.
#
# Usage:
#   1. Edit the COMMAND variable below to specify your training command
#   2. Submit with: qsub qsub_gpu_job.sh
#
# Examples of COMMAND values:
#   COMMAND="pixi run python src/genpp/train --config-name base_drm data=icon data.batch_size=8"
#   COMMAND="pixi run wandb agent feik/genpp/wgvukbrf"
#
# This script:
#   1. Creates a unique job directory on the local NVME storage (/raid/$USER/<job_id>)
#   2. Copies the ICON data from the shared SSD (/shared/data/$USER/icon) to local NVME
#   3. Sets the GENPP_DATA_DIR environment variable to point to the local data
#   4. Runs the specified command
#   5. Cleans up the job directory after the job completes (success or failure)
#
# Notes:
#   - Data is copied per-job to ensure isolation when multiple jobs run on the same node
#   - Cleanup happens automatically, even if the job is interrupted (via trap)
#   - The script uses rsync for efficient copying
#   - CUDA_VISIBLE_DEVICES is automatically set by the batch system
#   - If norm_stats files don't exist, first run may take 10-30 minutes to compute them
#   - Memory warning? Try reducing batch_size in your training command

#============================================
# NQSV Batch System Directives (gp_norm_dgx)
#============================================
#PBS -N genpp_gpu_job
#PBS -q gp_norm_dgx
#PBS -S /bin/bash
#PBS --gpunum-lhost=1
#PBS --cpunum-lhost=16
#PBS -l memsz_job=240gb
#PBS -l vmemsz_job=1Tb
#PBS -l vmemsz_prc=1Tb
#PBS -l elapstim_req=18:00:00
#PBS -j o
#PBS -o logs/train_%r.log

#============================================
# EDIT THIS: Specify your command here
#============================================
# Use -u flag for unbuffered Python output to see progress in real-time
COMMAND="pixi run -e gpu wandb agent feik/GenPP-src_genpp/z3bv09xb"

#============================================
# Do not edit below this line
#============================================

set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# Configuration
SOURCE_DATA_DIR="/shared/data/$USER/icon"
RAID_BASE_DIR="/raid"
USER_RAID_DIR="${RAID_BASE_DIR}/${USER}"

# Generate unique job ID using timestamp, PID, and random bytes for uniqueness
# Using 8 bytes of random data for better collision resistance in high-concurrency scenarios
JOB_ID="job_$(date +%Y%m%d_%H%M%S)_$$_$(head -c 8 /dev/urandom | xxd -p)"
JOB_DATA_DIR="${USER_RAID_DIR}/${JOB_ID}"

# Set WD
cd /hpc/uhome/$USER/GenPP

echo "=============================================="
echo "GenPP GPU Job (qsub)"
echo "=============================================="
echo "Job ID: ${JOB_ID}"
echo "PBS Job ID: ${PBS_JOBID:-N/A}"
echo "Source data: ${SOURCE_DATA_DIR}"
echo "Local data: ${JOB_DATA_DIR}"
echo "Command: ${COMMAND}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Current WD: $(pwd)"
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

# Create job-specific data directory with tensors subdirectory
# The source data has fc/ and rea/ directories, but the code expects them under tensors/
echo "Creating job data directory: ${JOB_DATA_DIR}"
mkdir -p "${JOB_DATA_DIR}/tensors"

# Copy data to local NVME storage
echo "Copying data to local NVME storage..."
echo "This may take a few minutes depending on data size..."
START_TIME=$(date +%s)

# Use --no-group and --no-owner to avoid unnecessary overhead when copying to local storage
# Copy into tensors/ subdirectory to match expected structure
if ! rsync -a --no-group --no-owner --info=STATS2 "${SOURCE_DATA_DIR}/" "${JOB_DATA_DIR}/tensors/"; then
    echo "ERROR: Failed to copy data from ${SOURCE_DATA_DIR} to ${JOB_DATA_DIR}/tensors"
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Data copy completed in ${DURATION} seconds"

# Verify critical files were copied
echo ""
echo "Verifying data structure..."
if [ -d "${JOB_DATA_DIR}/tensors/fc" ]; then
    echo "✓ Found tensors/fc/ directory"
    FC_FILE_COUNT=$(find "${JOB_DATA_DIR}/tensors/fc" -name "fc_*.pt" -type f | wc -l)
    echo "  FC tensor files: ${FC_FILE_COUNT}"
else
    echo "✗ ERROR: tensors/fc/ directory not found!"
fi

if [ -d "${JOB_DATA_DIR}/tensors/rea" ]; then
    echo "✓ Found tensors/rea/ directory"
    REA_FILE_COUNT=$(find "${JOB_DATA_DIR}/tensors/rea" -name "rea_*.pt" -type f | wc -l)
    echo "  REA tensor files: ${REA_FILE_COUNT}"
else
    echo "✗ ERROR: tensors/rea/ directory not found!"
fi

# Check for norm stats file (important for performance)
NORM_STATS_FILES=$(find "${JOB_DATA_DIR}/tensors" -maxdepth 1 -name "norm_stats_*.pt" -type f | wc -l)
if [ ${NORM_STATS_FILES} -gt 0 ]; then
    echo "✓ Found ${NORM_STATS_FILES} norm stats file(s) - will use cached statistics"
else
    echo "⚠ WARNING: No norm stats files found"
    echo "  The job will compute statistics from scratch, which may take 10-30 minutes"
    echo "  and use significant memory. Consider pre-computing norm stats in the source"
    echo "  directory to speed up job startup."
fi

# Check for feature metadata
if [ -f "${JOB_DATA_DIR}/tensors/fc/feature_metadata.pkl" ]; then
    echo "✓ Found feature_metadata.pkl"
else
    echo "✗ ERROR: feature_metadata.pkl not found!"
fi

# Export environment variable for the data directory
export GENPP_DATA_DIR="${JOB_DATA_DIR}"
echo ""
echo "Set GENPP_DATA_DIR=${GENPP_DATA_DIR}"

echo ""
echo "=============================================="
echo "Starting job..."
echo "=============================================="

# Run the provided command and capture exit status for informative error messages
set +e  # Disable exit on error temporarily to capture the exit code
eval "${COMMAND}"
CMD_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $CMD_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "ERROR: Command failed with exit code ${CMD_EXIT_CODE}"
    echo "Command: ${COMMAND}"
    echo "=============================================="
    exit $CMD_EXIT_CODE
fi

echo ""
echo "=============================================="
echo "Job completed successfully"
echo "=============================================="
