#!/bin/bash

#PBS -N merge_rea                                 # Job name
#PBS -S /bin/bash                                 # set the executing shell
#PBS -q rc_express                                # queue name
#PBS -l cpunum_job=1                              # use 1 CPU
#PBS -l memsz_job=2gb                             # total memory for job
#PBS -l vmemsz_job=2gb                            # total virtual memory
#PBS -l elapstim_req=00:30:00                     # max runtime: 30 minutes
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/merge_rea.log
#PBS -j o                                         # concatenate stderr and stdout

# This script processes reanalysis data on a per-month basis, creating one .nc file
# per (day, hour) combination. We only select the variables T_2M and VMAX_10M.
# To precompute the remap grid use get_remap_grid_rea.sh

module load cdo

# --- Read task parameters from task list using sub-request number ---
if [ -z "$TASK_LIST" ] || [ -z "$PBS_SUBREQNO" ]; then
    echo "ERROR: TASK_LIST and PBS_SUBREQNO must be set."
    echo "This script is meant to be submitted as an array job."
    exit 1
fi

TASK_LINE=$(sed -n "${PBS_SUBREQNO}p" "$TASK_LIST")
if [ -z "$TASK_LINE" ]; then
    echo "ERROR: No task found for sub-request number ${PBS_SUBREQNO}"
    exit 1
fi

YEAR=$(echo $TASK_LINE | awk '{print $1}')
MONTH=$(echo $TASK_LINE | awk '{print $2}')

echo "Sub-request ${PBS_SUBREQNO}: Processing ${YEAR}-${MONTH}"

# Paths
dataDir=/hpc/rwork2/evalpp/data/rea_grid_0037_R03B07
targetDomain=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/target_grid.txt
intWeights=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolation_weights/remap_gennn_rea.nc

# Output
output_dir=/hpc/uwork/extmfeik/data
output_folder=${output_dir}/rea
mkdir -p $output_folder

# Loop over all days in this month
for dayDir in $(ls -d ${dataDir}/${YEAR}/${MONTH}/*/ 2>/dev/null | sort); do
    DAY=$(basename $dayDir)

    # Loop over lead times 00, 12
    for HOUR in 00 12; do
        outFile=${output_folder}/rea_${YEAR}${MONTH}${DAY}${HOUR}.nc

        # Skip if already computed
        if [ -f "$outFile" ]; then
            echo "Skipping ${YEAR}-${MONTH}-${DAY} ${HOUR}h - already computed"
            continue
        fi

        # Get the GRIB file for this day and hour
        gribFile=$(find ${dataDir}/${YEAR}/${MONTH}/${DAY} -type f -name "*${YEAR}${MONTH}${DAY}${HOUR}.grib")

        if [ -z "$gribFile" ]; then
            echo "No file found for ${YEAR}-${MONTH}-${DAY} ${HOUR}h, skipping..."
            continue
        fi

        echo "Processing: $gribFile"

        # Process the file: select variables, remap
        cdo -f nc4 -z zip_6 \
            remap,$targetDomain,$intWeights \
            -selname,T_2M,VMAX_10M \
            $gribFile \
            $outFile

        echo "Done: $outFile"
    done
done

echo "Done processing ${YEAR}-${MONTH}!"
