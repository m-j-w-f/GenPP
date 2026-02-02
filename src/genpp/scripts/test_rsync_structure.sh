#!/bin/bash
# Test script to diagnose rsync directory structure issue
# This script copies a limited subset of files and shows the directory structure
# to help identify why the tensors directory is not being found

set -euo pipefail

# Configuration - same as qsub_gpu_job.sh
SOURCE_DATA_DIR="/shared/data/$USER/icon"
TEST_DIR="/tmp/test_rsync_$$"

echo "=============================================="
echo "RSYNC Directory Structure Test"
echo "=============================================="
echo "Source: ${SOURCE_DATA_DIR}"
echo "Test destination: ${TEST_DIR}"
echo ""

# Check if source exists
if [ ! -d "${SOURCE_DATA_DIR}" ]; then
    echo "ERROR: Source directory does not exist: ${SOURCE_DATA_DIR}"
    exit 1
fi

echo "=== SOURCE DIRECTORY STRUCTURE ==="
echo "Listing structure of ${SOURCE_DATA_DIR}:"
echo ""
# Show top-level structure
echo "Top-level contents:"
ls -lah "${SOURCE_DATA_DIR}/" 2>/dev/null || echo "Cannot list source directory"
echo ""

# Show subdirectories
echo "Directory tree (max depth 3, first 50 lines):"
tree -L 3 -d "${SOURCE_DATA_DIR}" 2>/dev/null | head -50 || \
    find "${SOURCE_DATA_DIR}" -maxdepth 3 -type d | head -50 || \
    echo "Cannot show tree structure"
echo ""

# Check for tensors directory
if [ -d "${SOURCE_DATA_DIR}/tensors" ]; then
    echo "✓ tensors/ directory EXISTS in source"
    echo "  Contents of tensors/:"
    ls -lah "${SOURCE_DATA_DIR}/tensors/" 2>/dev/null || echo "  Cannot list"
else
    echo "✗ tensors/ directory DOES NOT EXIST in source"
fi
echo ""

# Check for fc and rea directories
if [ -d "${SOURCE_DATA_DIR}/tensors/fc" ]; then
    echo "✓ tensors/fc/ directory EXISTS in source"
    echo "  Sample files (first 5):"
    ls "${SOURCE_DATA_DIR}/tensors/fc/" | head -5 || echo "  Cannot list"
    echo "  Total files in fc/:"
    find "${SOURCE_DATA_DIR}/tensors/fc/" -type f | wc -l
elif [ -d "${SOURCE_DATA_DIR}/fc" ]; then
    echo "✓ fc/ directory EXISTS in source (WITHOUT tensors/ parent)"
    echo "  Sample files (first 5):"
    ls "${SOURCE_DATA_DIR}/fc/" | head -5 || echo "  Cannot list"
    echo "  Total files in fc/:"
    find "${SOURCE_DATA_DIR}/fc/" -type f | wc -l
else
    echo "✗ Neither tensors/fc/ nor fc/ directory found in source"
fi
echo ""

if [ -d "${SOURCE_DATA_DIR}/tensors/rea" ]; then
    echo "✓ tensors/rea/ directory EXISTS in source"
    echo "  Sample files (first 5):"
    ls "${SOURCE_DATA_DIR}/tensors/rea/" | head -5 || echo "  Cannot list"
elif [ -d "${SOURCE_DATA_DIR}/rea" ]; then
    echo "✓ rea/ directory EXISTS in source (WITHOUT tensors/ parent)"
    echo "  Sample files (first 5):"
    ls "${SOURCE_DATA_DIR}/rea/" | head -5 || echo "  Cannot list"
else
    echo "✗ Neither tensors/rea/ nor rea/ directory found in source"
fi
echo ""

echo "=============================================="
echo "=== TESTING RSYNC ==="
echo "=============================================="

# Create test directory
mkdir -p "${TEST_DIR}/tensors"

echo "Test 1: Rsync with limited files (same command as qsub_gpu_job.sh)"
echo "Command: rsync -a --no-group --no-owner --max-size=1M --exclude='*.pt' \"${SOURCE_DATA_DIR}/\" \"${TEST_DIR}/tensors/\""
echo ""

# Do rsync with size limit and excluding large .pt files to make it fast
# NOTE: We copy into TEST_DIR/tensors/ to match the expected structure
rsync -a --no-group --no-owner --max-size=1M --exclude='*.pt' "${SOURCE_DATA_DIR}/" "${TEST_DIR}/tensors/" || {
    echo "Rsync failed!"
    exit 1
}

echo ""
echo "=== DESTINATION DIRECTORY STRUCTURE ==="
echo "Listing structure of ${TEST_DIR}:"
echo ""

echo "Top-level contents:"
ls -lah "${TEST_DIR}/" 2>/dev/null || echo "Cannot list destination"
echo ""

echo "Directory tree (max depth 3, first 50 lines):"
tree -L 3 -d "${TEST_DIR}" 2>/dev/null | head -50 || \
    find "${TEST_DIR}" -maxdepth 3 -type d | head -50 || \
    echo "Cannot show tree structure"
echo ""

# Check what was created
echo "=== CHECKING EXPECTED PATHS ==="
if [ -d "${TEST_DIR}/tensors" ]; then
    echo "✓ tensors/ directory EXISTS in destination"
    ls -lah "${TEST_DIR}/tensors/" 2>/dev/null || echo "  Cannot list"
else
    echo "✗ tensors/ directory DOES NOT EXIST in destination"
fi
echo ""

if [ -d "${TEST_DIR}/tensors/fc" ]; then
    echo "✓ tensors/fc/ directory EXISTS in destination"
    ls "${TEST_DIR}/tensors/fc/" | head -5 || echo "  Cannot list"
elif [ -d "${TEST_DIR}/fc" ]; then
    echo "✓ fc/ directory EXISTS in destination (WITHOUT tensors/ parent)"
    ls "${TEST_DIR}/fc/" | head -5 || echo "  Cannot list"
else
    echo "✗ Neither tensors/fc/ nor fc/ directory found in destination"
fi
echo ""

if [ -d "${TEST_DIR}/tensors/rea" ]; then
    echo "✓ tensors/rea/ directory EXISTS in destination"
    ls "${TEST_DIR}/tensors/rea/" | head -5 || echo "  Cannot list"
elif [ -d "${TEST_DIR}/rea" ]; then
    echo "✓ rea/ directory EXISTS in destination (WITHOUT tensors/ parent)"
    ls "${TEST_DIR}/rea/" | head -5 || echo "  Cannot list"
else
    echo "✗ Neither tensors/rea/ nor rea/ directory found in destination"
fi
echo ""

echo "=============================================="
echo "=== DIAGNOSIS ==="
echo "=============================================="

# Provide diagnosis
if [ -d "${SOURCE_DATA_DIR}/fc" ] && [ ! -d "${SOURCE_DATA_DIR}/tensors" ]; then
    echo "ISSUE IDENTIFIED:"
    echo "  Source has fc/ and rea/ directly under ${SOURCE_DATA_DIR}"
    echo "  But code expects tensors/fc/ and tensors/rea/"
    echo ""
    echo "SOLUTION APPLIED:"
    echo "  Updated rsync to copy INTO tensors/ subdirectory:"
    echo "  rsync ... \"\${SOURCE_DATA_DIR}/\" \"\${JOB_DATA_DIR}/tensors/\""
    echo ""
    if [ -d "${TEST_DIR}/tensors/fc" ]; then
        echo "✓ FIX VERIFIED: tensors/fc/ now exists in destination!"
    else
        echo "✗ FIX FAILED: tensors/fc/ still not found"
    fi
elif [ -d "${SOURCE_DATA_DIR}/tensors/fc" ]; then
    echo "Source structure looks correct (has tensors/fc/)"
    echo "The original rsync command should work in this case."
fi

echo ""
echo "=============================================="
echo "Cleaning up test directory: ${TEST_DIR}"
rm -rf "${TEST_DIR}"
echo "Done!"
