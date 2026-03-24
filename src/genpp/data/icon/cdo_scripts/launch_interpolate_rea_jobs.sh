#!/bin/bash

# Script to launch a single array job for all (year, month) combinations
# Each sub-request processes one month of reanalysis data

# Create log directory
SCRIPT_DIR=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts
mkdir -p ${SCRIPT_DIR}/logs

# Path to the processing script
SCRIPT_PATH=${SCRIPT_DIR}/interpolate_rea.sh

data_dir=/hpc/rwork2/evalpp/data/rea_grid_0037_R03B07

# Task list file: each line is "YEAR MONTH"
TASK_LIST=${SCRIPT_DIR}/rea_task_list.txt

# Generate the task list
> $TASK_LIST
for year in {2019..2022}; do
    for month in {01..12}; do

        # Skip future months for 2022 (adjust as needed)
        if [ $year -eq 2022 ] && [ $month -gt 11 ]; then
            continue
        fi

        # Only add if this year/month directory has data
        if [ -d "${data_dir}/${year}/${month}" ]; then
            echo "${year} ${month}" >> $TASK_LIST
        fi
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
