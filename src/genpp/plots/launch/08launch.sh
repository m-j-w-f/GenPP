#!/bin/bash -l
#PBS -N icon_pit_hist
#PBS -q rc_big
#PBS -S /bin/bash
#PBS -l cpunum_job=8
#PBS -l memsz_job=128gb
#PBS -l vmemsz_job=128gb
#PBS -l vmemsz_prc=128gb
#PBS -l elapstim_req=04:00:00
#PBS -j o
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/plots/launch/logs/icon_pit_histograms_%r.log

set -euo pipefail

REPO_ROOT=/hpc/uhome/extmfeik/GenPP
SCRIPT_PATH=${REPO_ROOT}/src/genpp/plots/08plot_histograms_icon.py

DEFAULT_OUTPUTS_ROOT=${REPO_ROOT}/outputs
DEFAULT_OBS_DIR=${REPO_ROOT}/src/genpp/data/icon/data/rea
DEFAULT_RESULTS_DIR=${REPO_ROOT}/outputs/results/icon/pit
DEFAULT_BASELINE_PATH=${REPO_ROOT}/outputs/BASELINE/test_predictions.nc

OUTPUTS_ROOT=${OUTPUTS_ROOT:-$DEFAULT_OUTPUTS_ROOT}
OBS_DIR=${OBS_DIR:-$DEFAULT_OBS_DIR}
RESULTS_DIR=${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}
BASELINE_PATH=${BASELINE_PATH:-$DEFAULT_BASELINE_PATH}

CHUNK_SIZE=${CHUNK_SIZE:-64}
BINS=${BINS:-40}
SEED=${SEED:-42}
MAX_TIMES=${MAX_TIMES:-}
DRY_RUN=${DRY_RUN:-0}
VERBOSE=${VERBOSE:-0}

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

mkdir -p ${REPO_ROOT}/src/genpp/plots/launch/logs
cd ${REPO_ROOT}

echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${PBS_JOBID:-N/A}"
echo "Script: ${SCRIPT_PATH}"
echo "Outputs root: ${OUTPUTS_ROOT}"
echo "Obs dir: ${OBS_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Baseline path: ${BASELINE_PATH}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "Bins: ${BINS}"
echo "Seed: ${SEED}"
echo "Max times: ${MAX_TIMES:-<all>}"
echo "Dry run: ${DRY_RUN}"
echo "Verbose: ${VERBOSE}"
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
EXTRA_ARGS+=" --outputs-root ${OUTPUTS_ROOT}"
EXTRA_ARGS+=" --obs-dir ${OBS_DIR}"
EXTRA_ARGS+=" --results-dir ${RESULTS_DIR}"
EXTRA_ARGS+=" --baseline-path ${BASELINE_PATH}"
EXTRA_ARGS+=" --chunk-size ${CHUNK_SIZE}"
EXTRA_ARGS+=" --bins ${BINS}"
EXTRA_ARGS+=" --seed ${SEED}"

if [ -n "${MAX_TIMES}" ]; then
    EXTRA_ARGS+=" --max-times ${MAX_TIMES}"
fi
if [ "${DRY_RUN}" = "1" ]; then
    EXTRA_ARGS+=" --dry-run"
fi
if [ "${VERBOSE}" = "1" ]; then
    EXTRA_ARGS+=" --verbose"
fi

COMMAND="pixi run python ${SCRIPT_PATH}${EXTRA_ARGS}"

echo "Running command: ${COMMAND}"
eval "${COMMAND}"
EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
