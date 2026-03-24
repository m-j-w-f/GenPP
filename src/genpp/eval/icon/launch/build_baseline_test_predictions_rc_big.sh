#!/bin/bash -l
#PBS -N baseline_test_pred
#PBS -q rc_big
#PBS -S /bin/bash
#PBS -l cpunum_job=8
#PBS -l memsz_job=96gb
#PBS -l vmemsz_job=96gb
#PBS -l vmemsz_prc=12gb
#PBS -l elapstim_req=06:00:00
#PBS -j o
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/eval/icon/launch/logs/baseline_test_predictions_%r.log

set -euo pipefail

REPO_ROOT=/hpc/uhome/extmfeik/GenPP
DEFAULT_REF=${REPO_ROOT}/outputs/ENGRESSION/2026-03-23_18-58-40/test_predictions.nc
DEFAULT_ENS=${REPO_ROOT}/src/genpp/data/icon/data/ens
DEFAULT_OUT=${REPO_ROOT}/outputs/BASELINE/test_predictions.nc

REF_TEST_PREDICTIONS=${REF_TEST_PREDICTIONS:-$DEFAULT_REF}
ENS_DIR=${ENS_DIR:-$DEFAULT_ENS}
OUTPUT_PATH=${OUTPUT_PATH:-$DEFAULT_OUT}
ALLOW_MISSING=${ALLOW_MISSING:-0}
OVERWRITE=${OVERWRITE:-1}

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

mkdir -p ${REPO_ROOT}/src/genpp/eval/icon/launch/logs
cd ${REPO_ROOT}

echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${PBS_JOBID:-N/A}"
echo "Reference: ${REF_TEST_PREDICTIONS}"
echo "Ens dir: ${ENS_DIR}"
echo "Output: ${OUTPUT_PATH}"
echo "=========================================="

if [ -f "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
elif [ -f "$HOME/.local/bin/pixi" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: pixi command not found in PATH"
    exit 1
fi

EXTRA_ARGS=""
if [ "$ALLOW_MISSING" = "1" ]; then
    EXTRA_ARGS+=" --allow-missing"
fi
if [ "$OVERWRITE" = "1" ]; then
    EXTRA_ARGS+=" --overwrite"
fi

COMMAND="pixi run python src/genpp/eval/icon/build_baseline_test_predictions.py --reference-test-predictions ${REF_TEST_PREDICTIONS} --ens-dir ${ENS_DIR} --output-path ${OUTPUT_PATH}${EXTRA_ARGS}"

echo "Running command: ${COMMAND}"
eval "${COMMAND}"
EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
