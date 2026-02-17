#!/bin/bash

# Launch monthly jobs to score raw ensemble forecasts (Energy Score + CRPS)

LOG_DIR=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs
SCRIPT_PATH=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/evaluate_raw_ensemble.sh

mkdir -p "${LOG_DIR}"

for year in {2018..2024}; do
    for month in {01..12}; do
        # Skip future months for 2024 (adjust as needed)
        if [ $year -eq 2024 ] && [ $month -gt 09 ]; then
            continue
        fi

        echo "Submitting raw ensemble scoring job for ${year}-${month}..."

        qsub -N icon_raw_${year}${month} \
             -o ${LOG_DIR}/raw_ensemble_${year}${month}.log \
             -v YEAR=${year},MONTH=${month} \
             ${SCRIPT_PATH}

        sleep 0.2
    done
done

echo ""
echo "All scoring jobs submitted."
echo "Monitor with: qstat -u $USER"
