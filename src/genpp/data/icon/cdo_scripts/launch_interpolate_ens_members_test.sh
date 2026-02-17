#!/bin/bash

# Script to launch a single test job for one day with all lead times.
# Useful for verifying the ensemble member extraction pipeline before
# submitting the full batch of monthly jobs.

# Create log directory
mkdir -p /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs

# Path to the main processing script
SCRIPT_PATH=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolate_ens_members.sh

# Test date: 2018-02-01
YEAR=2018
MONTH=02
DAY=01

echo "Submitting test job for ${YEAR}-${MONTH}-${DAY} (single day test)..."

qsub -N icon_ens_mem_test_${YEAR}${MONTH}${DAY} \
     -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/icon_ens_members_test_${YEAR}${MONTH}${DAY}.log \
     -v YEAR=${YEAR},MONTH=${MONTH},DAY=${DAY} \
     ${SCRIPT_PATH}

echo ""
echo "Test job submitted!"
echo "Monitor with: qstat -u $USER"
