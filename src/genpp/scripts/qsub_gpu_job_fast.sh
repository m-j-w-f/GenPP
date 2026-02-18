#!/bin/bash
# Submit this script with qsub for FAST testing using data directly from shared storage.
#
# This script is designed for the NQSV batch system and cannot accept command-line arguments.
# Instead, edit the COMMAND variable below before submitting with qsub.
#
# Usage:
#   1. Edit the COMMAND variable below to specify your training command
#   2. Submit with: qsub qsub_gpu_job_fast.sh
#
# Examples of COMMAND values:
#   COMMAND="pixi run python src/genpp/train --config-name base_drm data=icon data.batch_size=8"
#   COMMAND="pixi run wandb agent feik/genpp/wgvukbrf"
#
# This script (FAST VERSION):
#   1. Uses data DIRECTLY from shared storage (/shared/data/$USER/icon)
#   2. Sets the GENPP_DATA_DIR environment variable to point to shared data
#   3. Runs the specified command
#   4. NO data copying - faster startup, but may have slower I/O during training
#
# Notes:
#   - FAST: No data copying means instant startup (saves 10-15 minutes)
#   - SLOWER I/O: Reading from shared storage may be slower than local NVME during training
#   - SHARED: Multiple jobs share the same data location
#   - USE FOR: Quick tests, debugging, small experiments
#   - For production runs, use qsub_gpu_job.sh (with NVME caching)
#   - CUDA_VISIBLE_DEVICES is automatically set by the batch system
#   - If norm_stats files don't exist, first run may take 10-30 minutes to compute them
#   - Memory warning? Try reducing batch_size in your training command

#============================================
# NQSV Batch System Directives (gp_norm_dgx)
#============================================
#PBS -N genpp_gpu_job_fast
#PBS -q gp_norm_smc
#PBS -S /bin/bash
#PBS --gpunum-lhost=1
#PBS --cpunum-lhost=16
#PBS -l memsz_job=240gb
#PBS -l vmemsz_job=1Tb
#PBS -l vmemsz_prc=1Tb
#PBS -l elapstim_req=00:30:00
#PBS -j o
#PBS -o logs/eval_fast_%r.log

#============================================
# EDIT THIS: Specify your command here
#============================================
# Evaluation

# EMOS
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_copulas_eval.py --run-path feik/genpp/3zggrfqs --split test -v --save-predictions --batch-size 4 --skip-variogram"

# DRN
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_copulas_eval.py --run-path feik/genpp/db1bgpg5 --split test -v --save-predictions --batch-size 4 --skip-variogram"

# LNGM
# LNGM (MSPES)
#COMMAND="pixi run -e gpu python src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/j2rg4w0o --split test -v --save-predictions --batch-size 4"
# LNGM (MSES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/5wv59jka --split test -v --save-predictions --batch-size 4 --skip-variogram"
# LNGM (PES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/rc4yel5e --split test -v --save-predictions --batch-size --skip-variogram4"
# LNGM (ES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/fngro7wf --split test -v --save-predictions --batch-size 4 --skip-variogram"

# ALL LNGM
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/j2rg4w0o feik/genpp/rc4yel5e feik/genpp/fngro7wf --split test -v --save-predictions --batch-size 4 --skip-variogram"

# Engression
# ENG (ES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/9o3mnwa8 --split test -v --save-predictions --batch-size 4 --skip-variogram"
# ENG (PES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/7pm11esx --split test -v --save-predictions --batch-size 4 --skip-variogram"
# ENG (MSPES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/xzafsu8a --split test -v --save-predictions --batch-size 4 --skip-variogram"
# ENG (MSES)
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/2xbli9p2 --split test -v --save-predictions --batch-size 4 --skip-variogram"

# ALL Engression
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/9o3mnwa8 feik/genpp/7pm11esx feik/genpp/xzafsu8a --split test -v --save-predictions --batch-size 4 --skip-variogram"

# FM
# UNET - IND
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/ibbb3wdk --split test -v --save-predictions --batch-size 4 --skip-variogram"
# UNET - DIR
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/38tym6f0 --split test -v --save-predictions --batch-size 4 --skip-variogram"
# UViT - IND
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/zo2uhaev --split test -v --save-predictions --batch-size 4 --skip-variogram"
# UViT - DIR
#COMMAND="pixi run -e gpu python -u src/genpp/eval/icon_predict_eval.py --run-path feik/genpp/9au1bayh --split test -v --save-predictions --batch-size 4 --skip-variogram"


#============================================
# Do not edit below this line
#============================================

set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1

# Configuration - use shared data directly
SHARED_DATA_DIR="/shared/data/$USER/icon"

# Set WD
cd /hpc/uhome/$USER/GenPP

echo "=============================================="
echo "GenPP GPU Job - FAST (Direct Shared Access)"
echo "=============================================="
echo "PBS Job ID: ${PBS_JOBID:-N/A}"
echo "Data source: ${SHARED_DATA_DIR} (direct access - no copying)"
echo "Command: ${COMMAND}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Current WD: $(pwd)"
echo "=============================================="

# Verify source data exists
if [ ! -d "${SHARED_DATA_DIR}" ]; then
    echo "ERROR: Shared data directory does not exist: ${SHARED_DATA_DIR}"
    echo "Please ensure the ICON data is available at the expected location."
    exit 1
fi

# Verify data structure in shared storage
echo ""
echo "Verifying shared data structure..."
if [ -d "${SHARED_DATA_DIR}/fc" ]; then
    echo "✓ Found fc/ directory in shared storage"
    FC_FILE_COUNT=$(find "${SHARED_DATA_DIR}/fc" -name "fc_*.pt" -type f 2>/dev/null | wc -l)
    echo "  FC tensor files: ${FC_FILE_COUNT}"
else
    echo "✗ ERROR: fc/ directory not found in shared storage!"
    exit 1
fi

if [ -d "${SHARED_DATA_DIR}/rea" ]; then
    echo "✓ Found rea/ directory in shared storage"
    REA_FILE_COUNT=$(find "${SHARED_DATA_DIR}/rea" -name "rea_*.pt" -type f 2>/dev/null | wc -l)
    echo "  REA tensor files: ${REA_FILE_COUNT}"
else
    echo "✗ ERROR: rea/ directory not found in shared storage!"
    exit 1
fi

# Check for norm stats file (important for performance)
NORM_STATS_FILES=$(find "${SHARED_DATA_DIR}" -maxdepth 1 -name "norm_stats_*.pt" -type f 2>/dev/null | wc -l)
if [ ${NORM_STATS_FILES} -gt 0 ]; then
    echo "✓ Found ${NORM_STATS_FILES} norm stats file(s) - will use cached statistics"
else
    echo "⚠ WARNING: No norm stats files found in shared storage"
    echo "  The job will compute statistics from scratch, which may take 10-30 minutes"
    echo "  and use significant memory. Consider pre-computing norm stats."
fi

# Check for feature metadata
if [ -f "${SHARED_DATA_DIR}/fc/feature_metadata.pkl" ]; then
    echo "✓ Found feature_metadata.pkl"
else
    echo "✗ ERROR: feature_metadata.pkl not found in shared storage!"
    exit 1
fi

# Create a temporary tensors symlink structure to match code expectations
# The code expects data_dir/tensors/fc and data_dir/tensors/rea
# But shared storage has fc/ and rea/ directly
# Solution: Create a temporary directory with symlinks
TEMP_DATA_DIR="/tmp/genpp_data_${USER}_$$"
mkdir -p "${TEMP_DATA_DIR}/tensors"

echo ""
echo "Creating temporary directory structure..."
echo "Temp dir: ${TEMP_DATA_DIR}"

# Create symlinks to maintain directory structure without copying data
ln -s "${SHARED_DATA_DIR}/fc" "${TEMP_DATA_DIR}/tensors/fc"
ln -s "${SHARED_DATA_DIR}/rea" "${TEMP_DATA_DIR}/tensors/rea"

# Symlink any norm_stats files
for stats_file in "${SHARED_DATA_DIR}"/norm_stats_*.pt; do
    if [ -f "${stats_file}" ]; then
        ln -s "${stats_file}" "${TEMP_DATA_DIR}/tensors/$(basename "${stats_file}")"
        echo "  Linked: $(basename "${stats_file}")"
    fi
done

echo "✓ Temporary structure created with symlinks (no data copying)"

# Cleanup function to remove temporary directory
cleanup() {
    local exit_code=$?
    echo ""
    echo "=============================================="
    echo "Cleaning up temporary directory..."
    if [ -d "${TEMP_DATA_DIR}" ]; then
        rm -rf "${TEMP_DATA_DIR}"
        echo "Removed: ${TEMP_DATA_DIR}"
    else
        echo "Temporary directory already cleaned up or doesn't exist"
    fi
    echo "=============================================="
    exit $exit_code
}

# Set trap to ensure cleanup runs on exit (success, failure, or interrupt)
trap cleanup EXIT INT TERM

# Export environment variable for the data directory (point to temp dir with symlinks)
# Note: Code expects data at GENPP_DATA_DIR/tensors/{fc,rea}
# We set GENPP_DATA_DIR to the parent directory containing the tensors/ subdirectory
export GENPP_DATA_DIR="${TEMP_DATA_DIR}"
echo ""
echo "Set GENPP_DATA_DIR=${GENPP_DATA_DIR}"
echo "  (uses symlinks to ${SHARED_DATA_DIR})"

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
