#!/bin/bash

# Script to launch a single array job for all (year, month, leadtime) combinations
# Each sub-request processes one month of data for a single leadtime

# Create log directory
SCRIPT_DIR=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts
mkdir -p ${SCRIPT_DIR}/logs

# Path to the main processing script
SCRIPT_PATH=${SCRIPT_DIR}/interpolate_ens.sh

# Task list file: each line is "YEAR MONTH LEADTIME"
TASK_LIST=${SCRIPT_DIR}/ens_task_list.txt

# Generate the task list
> $TASK_LIST
for year in {2019..2022}; do
    for month in {01..12}; do

        # Skip future months for 2024 (adjust as needed)
        if [ $year -eq 2022 ] && [ $month -gt 11 ]; then
            continue
        fi

        for leadtime in 12 24 36 48 60 72 84 96 108 120; do
            echo "${year} ${month} ${leadtime}" >> $TASK_LIST
        done
    done
done

# Count the number of tasks
NUM_TASKS=$(wc -l < $TASK_LIST)
echo "Generated ${NUM_TASKS} tasks in ${TASK_LIST}"

# Submit as a single array job with sub-request numbers 1 to NUM_TASKS
qsub -t 1-${NUM_TASKS} \
     -v TASK_LIST=${TASK_LIST} \
     ${SCRIPT_PATH}

echo ""
echo "Array job submitted with ${NUM_TASKS} sub-requests!"
echo "Monitor with: qstat -u $USER"
