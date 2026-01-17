#!/bin/bash -l

#PBS -N process_rea_tensors           # Job name (will be modified per month)
#PBS -S /bin/bash                      # set the executing shell
#PBS -l cpunum_job=4                   # use 4 CPUs
#PBS -l memsz_job=32gb                 # total memory for job
#PBS -l elapstim_req=12:00:00          # max runtime: 12 hours (per month)
#PBS -j o                              # concatenate stderr and stdout

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

# Set environment variables for the script
export JOB_TYPE=rea

# Change to the working directory from which qsub was called
# This assumes you run qsub from src/genpp/data/icon directory
cd "$PBS_O_WORKDIR" || exit 1

# Run the Python script using pixi
pixi run -e nb python process_tensors.py

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
