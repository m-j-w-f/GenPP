#!/bin/bash -l

#PBS -N process_fc_tensors           # Job name (will be modified per month)
#PBS -S /bin/bash                     # set the executing shell
#PBS -q rc_big
#PBS -l cpunum_job=4              # use 4 CPUs (for CDO OpenMP)
#PBS -l memsz_job=8gb            # total memory for job
#PBS -l vmemsz_job=8gb           # total virtual memory
#PBS -l elapstim_req=01:00:00         # max runtime: 1 hour (per month)
#PBS -j o                             # concatenate stderr and stdout

# Year and Month should be env vars passed via qsub -v
if [ -z "$YEAR" ] || [ -z "$MONTH" ]; then
    echo "ERROR: Year and month required as environment variables"
    echo "Usage: qsub -v YEAR=YYYY,MONTH=MM $0"
    exit 1
fi

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Processing year: $YEAR, month: $MONTH"
echo "=========================================="

# Export environment variables for the Python script
export YEAR
export MONTH
export JOB_TYPE=fc

# Change to the working directory from which qsub was called
# This assumes you run qsub from src/genpp/data/icon directory
cd "$PBS_O_WORKDIR" || exit 1

# Add pixi to PATH if not already present
# Common installation locations for pixi
if [ -f "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
elif [ -f "$HOME/.local/bin/pixi" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify pixi is available
if ! command -v pixi &> /dev/null; then
    echo "ERROR: pixi command not found in PATH"
    echo "Please ensure pixi is installed and accessible"
    exit 1
fi

echo "Using pixi: $(which pixi)"

# Run the Python script using pixi
pixi run -e nb python process_tensors.py

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
