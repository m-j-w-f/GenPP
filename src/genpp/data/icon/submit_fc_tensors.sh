#!/bin/bash
#PBS -N process_fc_tensors
#PBS -l elapstim_req=12:00:00
#PBS -l cpunum_job=4
#PBS -l memsz_job=32gb
#PBS -t 1-24
#PBS -o logs/fc_tensor_%t.out
#PBS -e logs/fc_tensor_%t.err
#PBS -j o

# Array job for processing forecast (FC) tensors in parallel
# Each array task processes data for one month
# Adjust the -t parameter to match the number of months you want to process

# Example months (modify as needed based on your data range)
# This maps array task ID to year-month
declare -a MONTHS=(
    "2019-01" "2019-02" "2019-03" "2019-04" "2019-05" "2019-06"
    "2019-07" "2019-08" "2019-09" "2019-10" "2019-11" "2019-12"
    "2020-01" "2020-02" "2020-03" "2020-04" "2020-05" "2020-06"
    "2020-07" "2020-08" "2020-09" "2020-10" "2020-11" "2020-12"
)

# Get the year-month for this task based on PBS array task ID
TASK_ID=$PBS_SUBREQNO
YEAR_MONTH=${MONTHS[$((TASK_ID-1))]}

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Task ID: $TASK_ID"
echo "Processing month: $YEAR_MONTH"
echo "=========================================="

# Set environment variables for the script
export JOB_TYPE=fc
export YEAR_MONTH=$YEAR_MONTH

# Change to the script directory
cd /home/runner/work/GenPP/GenPP/src/genpp/data/icon || exit 1

# Run the Python script using pixi
pixi run -e nb python process_tensors.py

EXIT_CODE=$?

echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
