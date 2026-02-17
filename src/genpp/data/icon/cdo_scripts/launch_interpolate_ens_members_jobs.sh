#!/bin/bash

# Script to launch parallel jobs for each month to save ensemble members
# Each job processes one month of data

# Create log directory
mkdir -p /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs

# Path to the main processing script
SCRIPT_PATH=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolate_ens_members.sh

# Loop through years and months
for year in {2018..2024}; do
    for month in {01..12}; do

        # Skip future months for 2024 (adjust as needed)
        if [ $year -eq 2024 ] && [ $month -gt 09 ]; then
            continue
        fi

        echo "Submitting job for ${year}-${month}..."

        # Submit the job with year and month as arguments
        # Use -N to set unique job name
        qsub -N icon_ens_mem_${year}${month} \
             -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/icon_ens_members_${year}${month}.log \
             -v YEAR=${year},MONTH=${month} \
             ${SCRIPT_PATH}

        # Optional: add a small delay to avoid overwhelming the scheduler
        sleep 0.2
    done
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
