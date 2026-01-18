#!/bin/bash

# Script to launch parallel jobs for tensor processing
# Each job processes one month of data

# Create log directory
mkdir -p logs

# Path to the processing scripts
FC_SCRIPT_PATH="$(pwd)/submit_fc_tensors.sh"
REA_SCRIPT_PATH="$(pwd)/submit_rea_tensors.sh"

# Loop through years and months
for year in {2018..2024}; do
    for month in {01..12}; do

        # Skip future months for 2024 (adjust as needed)
        if [ $year -eq 2024 ] && [ $month -gt 09 ]; then
            continue
        fi

        echo "Submitting jobs for ${year}-${month}..."

        # Submit FC tensor processing job
        qsub -N fc_tensor_${year}${month} \
             -o logs/fc_tensor_${year}${month}.log \
             -v YEAR=${year},MONTH=${month} \
             ${FC_SCRIPT_PATH}

        # Submit REA tensor processing job
        qsub -N rea_tensor_${year}${month} \
             -o logs/rea_tensor_${year}${month}.log \
             -v YEAR=${year},MONTH=${month} \
             ${REA_SCRIPT_PATH}
    done
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
