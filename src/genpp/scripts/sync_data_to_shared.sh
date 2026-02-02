#!/bin/bash
#PBS -N sync_data_to_shared
#PBS -q gp_norm_dgx
#PBS -S /bin/bash
#PBS --gpunum-lhost=0
#PBS --cpunum-lhost=8
#PBS -l memsz_job=32gb
#PBS -l vmemsz_job=32gb
#PBS -l vmemsz_prc=32gb
#PBS -l elapstim_req=01:00:00
#PBS -j o
#PBS -o sync_data_to_shared.log
# Sync local ICON data to the shared storage on GPU nodes.
#
# This script must be run on a GPU node as only these nodes have access to /shared.
# No GPU is required for this operation.
#
# This script:
#   1. Copies data from src/genpp/data/icon/data/tensors to /shared/data/$USER/icon
#      (i.e., the contents of the tensors directory: fc/ and rea/ subdirectories)
#   2. Verifies the transfer by comparing file counts and sizes
#   3. Produces output to verify the transaction worked
#
# Note: The qsub_gpu_job.sh script will copy this data back into a tensors/ subdirectory
#       to match the expected structure: ${JOB_DATA_DIR}/tensors/fc/ and ${JOB_DATA_DIR}/tensors/rea/
#
# Usage:
#   ./sync_data_to_shared.sh
#
# To run on a GPU node without a GPU allocation:
# qsub sync_data_to_shared.sh
#
# Notes:
#   - This script should be run before using launch_gpu_job.sh or qsub_gpu_job.sh
#   - The shared storage is available at /shared/data with 134TB capacity
#   - Uses rsync for efficient, resumable transfers

set -euo pipefail

SOURCE_DATA_DIR="${HOME}/GenPP/src/genpp/data/icon/data/tensors"

# Destination on shared storage
DEST_DATA_DIR="/shared/data/${USER}/icon"

echo "=============================================="
echo "GenPP Data Sync to Shared Storage"
echo "=============================================="
echo "Source:      ${SOURCE_DATA_DIR}"
echo "Destination: ${DEST_DATA_DIR}"
echo "=============================================="

# Verify source data exists
if [ ! -d "${SOURCE_DATA_DIR}" ]; then
    echo "ERROR: Source data directory does not exist: ${SOURCE_DATA_DIR}"
    echo "Please ensure the ICON data is available at the expected location."
    exit 1
fi

# Check if /shared is accessible (only available on GPU nodes)
if [ ! -d "/shared/data" ]; then
    echo "ERROR: /shared/data is not accessible."
    echo "This script must be run on a GPU node (smc or dgx)."
    echo ""
    echo "To get an interactive session on a GPU node without GPU allocation:"
    echo "  qlogin -q gp_inter_dgx --cpunum-lhost=1 -l memsz_job=15gb -l vmemsz_job=15gb -l vmemsz_prc=15gb"
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "${DEST_DATA_DIR}" ]; then
    echo "Creating destination directory: ${DEST_DATA_DIR}"
    mkdir -p "${DEST_DATA_DIR}"
fi

# Get source statistics before copy
echo ""
echo "Analyzing source data..."
SOURCE_FILE_COUNT=$(find "${SOURCE_DATA_DIR}" -type f | wc -l)
SOURCE_DIR_COUNT=$(find "${SOURCE_DATA_DIR}" -type d | wc -l)
SOURCE_SIZE=$(du -sh "${SOURCE_DATA_DIR}" 2>/dev/null | cut -f1)
echo "Source: ${SOURCE_FILE_COUNT} files, ${SOURCE_DIR_COUNT} directories, ${SOURCE_SIZE} total"

# Copy data to shared storage
echo ""
echo "=============================================="
echo "Starting data transfer..."
echo "=============================================="
START_TIME=$(date +%s)

# Use rsync with progress and archive mode for efficient transfer
# Note: --delete is intentionally omitted to avoid removing files that may have been added to shared storage
rsync -av --info=progress2 "${SOURCE_DATA_DIR}/" "${DEST_DATA_DIR}/"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Transfer completed in ${DURATION} seconds"
echo "=============================================="

# Verify the transfer
echo ""
echo "Verifying transfer..."
DEST_FILE_COUNT=$(find "${DEST_DATA_DIR}" -type f | wc -l)
DEST_DIR_COUNT=$(find "${DEST_DATA_DIR}" -type d | wc -l)
DEST_SIZE=$(du -sh "${DEST_DATA_DIR}" 2>/dev/null | cut -f1)

echo ""
echo "=============================================="
echo "Verification Results"
echo "=============================================="
echo "Source:      ${SOURCE_FILE_COUNT} files, ${SOURCE_DIR_COUNT} directories, ${SOURCE_SIZE}"
echo "Destination: ${DEST_FILE_COUNT} files, ${DEST_DIR_COUNT} directories, ${DEST_SIZE}"

# Compare counts
if [ "${SOURCE_FILE_COUNT}" -eq "${DEST_FILE_COUNT}" ] && [ "${SOURCE_DIR_COUNT}" -eq "${DEST_DIR_COUNT}" ]; then
    echo ""
    echo "✓ SUCCESS: File and directory counts match!"
    echo ""
    echo "Data is now available at: ${DEST_DATA_DIR}"
    echo "You can now use launch_gpu_job.sh or qsub_gpu_job.sh for training."
else
    echo ""
    echo "⚠ WARNING: File or directory counts do not match!"
    echo "  Source files: ${SOURCE_FILE_COUNT}, Destination files: ${DEST_FILE_COUNT}"
    echo "  Source dirs:  ${SOURCE_DIR_COUNT}, Destination dirs:  ${DEST_DIR_COUNT}"
    echo ""
    echo "Consider running this script again or investigating the differences."
    exit 1
fi

echo "=============================================="
