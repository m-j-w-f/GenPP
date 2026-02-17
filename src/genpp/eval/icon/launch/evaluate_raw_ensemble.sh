#!/bin/bash -l

#PBS -N icon_raw_scores            # Job name (will be modified per month)
#PBS -S /bin/bash                  # set the executing shell
#PBS -q rc_big
#PBS -l cpunum_job=4               # CPUs for torch/xarray work
#PBS -l memsz_job=16gb             # total memory for job
#PBS -l vmemsz_job=16gb            # total virtual memory
#PBS -l elapstim_req=02:00:00      # max runtime per month
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/eval/icon/launch/logs/raw_ensemble_${YEAR}${MONTH}.log
#PBS -j o                          # concatenate stderr and stdout

# Required env vars: YEAR (YYYY), MONTH (MM)
# Optional env vars: DAY (DD), LEADTIMES (comma-separated hours),
#                    DATA_DIR, OUTPUT_DIR

if [ -z "$YEAR" ] || [ -z "$MONTH" ]; then
    echo "ERROR: Year and month required as environment variables"
    echo "Usage: qsub -v YEAR=YYYY,MONTH=MM[,DAY=DD][,LEADTIMES=24,48] $0"
    exit 1
fi

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Processing year: $YEAR, month: $MONTH${DAY:+, day: $DAY}"
echo "=========================================="

# Change to the submission directory
cd "$PBS_O_WORKDIR" || exit 1

# Locate repository root (where pyproject.toml lives)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ]; do
    if [ -f "$REPO_ROOT/pyproject.toml" ]; then
        break
    fi
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done

if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
    echo "ERROR: Could not find pyproject.toml in parent directories"
    exit 1
fi

# Add pixi to PATH if available
if [ -f "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
elif [ -f "$HOME/.local/bin/pixi" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v pixi &> /dev/null; then
    echo "ERROR: pixi command not found in PATH"
    echo "Please ensure pixi is installed and accessible"
    exit 1
fi

# Prepare optional arguments
extra_args=""
if [ -n "$DAY" ]; then
    extra_args+=" --day ${DAY}"
fi

if [ -n "$LEADTIMES" ]; then
    IFS=',' read -ra lt_arr <<< "$LEADTIMES"
    for lt in "${lt_arr[@]}"; do
        extra_args+=" --leadtime ${lt}"
    done
fi

if [ -n "$DATA_DIR" ]; then
    extra_args+=" --data-dir ${DATA_DIR}"
fi

if [ -n "$OUTPUT_DIR" ]; then
    extra_args+=" --output-dir ${OUTPUT_DIR}"
fi

# Ensure log directory exists
mkdir -p /hpc/uhome/extmfeik/GenPP/src/genpp/eval/icon/launch/logs

cd "$REPO_ROOT" || exit 1
eval "$(pixi shell-hook)"
pixi run python src/genpp/eval/icon/raw_ensemble.py --year "$YEAR" --month "$MONTH" ${extra_args}
EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
