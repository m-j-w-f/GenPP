#!/bin/bash

# Submit a single test job to score one day of raw ensembles before launching all months.

LOG_DIR=/hpc/uhome/extmfeik/GenPP/src/genpp/eval/icon/launch/logs
SCRIPT_PATH=/hpc/uhome/extmfeik/GenPP/src/genpp/eval/icon/launch/evaluate_raw_ensemble.sh

mkdir -p "${LOG_DIR}"

# Use a representative early day with all lead times
YEAR=2018
MONTH=02
DAY=01

echo "Submitting raw ensemble scoring test for ${YEAR}-${MONTH}-${DAY}..."

qsub -N icon_raw_test_${YEAR}${MONTH}${DAY} \
     -o ${LOG_DIR}/raw_ensemble_test_${YEAR}${MONTH}${DAY}.log \
     -v YEAR=${YEAR},MONTH=${MONTH},DAY=${DAY} \
     ${SCRIPT_PATH}

echo "Done. Monitor with: qstat -u $USER"
