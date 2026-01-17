#!/bin/bash

# Script to launch parallel jobs for each day to process reanalysis data
# Each job processes one day of data

# Create log directory
mkdir -p /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs

# Path to the processing script
SCRIPT_PATH=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolate_rea.sh

data_dir=/hpc/rwork2/evalpp/data/rea_grid_0037_R03B07

# Loop through years and months
for year in {2018..2024}; do
    for month in {01..12}; do

        # Skip future months for 2024 (adjust as needed)
        if [ $year -eq 2024 ] && [ $month -gt 09 ]; then
            continue
        fi

        # Change to the month directory
        cd ${data_dir}/${year}/${month}

        # Get the list of day directories
        for day in $(ls -d */ 2>/dev/null | sort | sed 's|/||'); do
            if [ -z "$day" ]; then
                continue
            fi

            echo "Submitting job for ${year}-${month}-${day}..."

            # Submit the job with year, month, and day as arguments
            # Use -N to set unique job name
            qsub -N merge_rea_${year}${month}${day} \
                 -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/merge_rea_${year}${month}${day}.log \
                 -v YEAR=${year},MONTH=${month},DAY=${day} \
                 ${SCRIPT_PATH}

            # Optional: add a small delay to avoid overwhelming the scheduler
            sleep 0.01
        done

        # Go back to the original directory
        cd -
    done
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"