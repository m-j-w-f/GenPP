#!/bin/bash

# Script to launch array jobs for tensor processing
# Each sub-request processes one (job_type, year, month, day) combination
# Tasks are split into batches of up to 500 sub-requests per array job

# Create log directory
SCRIPT_DIR=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/scripts
mkdir -p ${SCRIPT_DIR}/logs

# Path to the processing script
SCRIPT_PATH=${SCRIPT_DIR}/submit_tensors.sh

# Task list file: each line is "JOB_TYPE YEAR MONTH DAY"
TASK_LIST=${SCRIPT_DIR}/tensor_task_list.txt

# Max sub-requests per array job
BATCH_SIZE=250

# Data directories for enumerating days
ens_data_dir=/hpc/uwork/extmfeik/data/ensmean
rea_data_dir=/hpc/uwork/extmfeik/data/rea

# Generate the task list
> $TASK_LIST
for year in {2019..2022}; do
    for month in {01..12}; do

        # Skip months after November 2022
        if [ $year -eq 2022 ] && [ $month -gt 11 ]; then
            continue
        fi

        # Get unique days from ensmean files for FC tasks
        fc_days=$(ls ${ens_data_dir}/ensmean_${year}${month}*.nc 2>/dev/null | \
            sed 's|.*/ensmean_||;s|\.nc||' | \
            cut -c7-8 | sort -u)
        for day in $fc_days; do
            echo "fc ${year} ${month} ${day}" >> $TASK_LIST
        done

        # Get unique days from rea files for REA tasks
        rea_days=$(ls ${rea_data_dir}/rea_${year}${month}*.nc 2>/dev/null | \
            sed 's|.*/rea_||;s|\.nc||' | \
            cut -c7-8 | sort -u)
        for day in $rea_days; do
            echo "rea ${year} ${month} ${day}" >> $TASK_LIST
        done
    done
done

# Count the number of tasks
NUM_TASKS=$(wc -l < $TASK_LIST)
echo "Generated ${NUM_TASKS} tasks in ${TASK_LIST}"

if [ "$NUM_TASKS" -eq 0 ]; then
    echo "No tasks to submit!"
    exit 0
fi

# Submit in batches of BATCH_SIZE
batch_num=0
for start in $(seq 1 $BATCH_SIZE $NUM_TASKS); do
    end=$((start + BATCH_SIZE - 1))
    if [ $end -gt $NUM_TASKS ]; then
        end=$NUM_TASKS
    fi
    batch_num=$((batch_num + 1))

    echo "Submitting batch ${batch_num}: sub-requests ${start}-${end}"
    qsub -t ${start}-${end} \
         -v TASK_LIST=${TASK_LIST} \
         ${SCRIPT_PATH}
done

echo ""
echo "Submitted ${batch_num} array jobs covering ${NUM_TASKS} tasks (batch size: ${BATCH_SIZE})"
echo "Monitor with: qstat -u $USER"
