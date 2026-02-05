#!/bin/bash

#PBS -N merge_rea_${YEAR}${MONTH}${DAY}           # Job name (will be modified per day)
#PBS -S /bin/bash                                 # set the executing shell
#PBS -q rc_express                                # queue name
#PBS -l cpunum_job=1                              # use 1 CPU
#PBS -l memsz_job=2gb                             # total memory for job
#PBS -l vmemsz_job=2gb                            # total virtual memory
#PBS -l elapstim_req=00:02:00                     # max runtime: 2 minutes
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/merge_rea_${YEAR}${MONTH}${DAY}.log
#PBS -j o                                         # concatenate stderr and stdout

# This script processes reanalysis data on a per-day basis, creating one .nc file per day.
# We only select the variables T_2M and VMAX_10M
# To precompute the remap grid use get_remap_grid_rea.sh

module load cdo

# Paths
dataDir=/hpc/rwork2/evalpp/data/rea_grid_0037_R03B07
targetDomain=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/target_grid.txt
intWeights=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolation_weights/remap_gennn_rea.nc

# Output
output_dir=/hpc/uwork/extmfeik/data
output_folder=${output_dir}/rea
mkdir -p $output_folder

# Year, Month, and Day should be env vars
if [ -z "$YEAR" ] || [ -z "$MONTH" ] || [ -z "$DAY" ]; then
    echo "ERROR: Year, month, and day required as arguments"
    echo "Usage: $0 YYYY MM DD"
    exit 1
fi

echo "Processing ${YEAR}-${MONTH}-${DAY}"

# Construct the date string
dateStr=${YEAR}${MONTH}${DAY}

# Get the GRIB file for this day (00 hour only)
gribFile=$(find ${dataDir}/${YEAR}/${MONTH}/${DAY} -type f -name "*${dateStr}00.grib")

if [ -z "$gribFile" ]; then
    echo "No file found for ${YEAR}-${MONTH}-${DAY}, skipping..."
    exit 0
fi

echo "Found file: $gribFile"

# Process the file: select variables, remap
cdo -f nc4 -z zip_6 \
    remap,$targetDomain,$intWeights \
    -selname,T_2M,VMAX_10M \
    $gribFile \
    ${output_folder}/rea_${YEAR}${MONTH}${DAY}.nc

echo "Done: ${output_folder}/rea_${YEAR}${MONTH}${DAY}.nc"