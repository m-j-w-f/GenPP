#!/bin/bash -l

#PBS -N icon_eu_eps_ens            # Job name
#PBS -S /bin/bash                  # set the executing shell
#PBS -q rc_big                     # queue name
#PBS -l cpunum_job=4              # use 4 CPUs (for CDO OpenMP)
#PBS -l memsz_job=8gb            # total memory for job
#PBS -l vmemsz_job=8gb           # total virtual memory
#PBS -l elapstim_req=01:00:00     # max runtime: 1 hours
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/icon_ens.log
#PBS -j o                          # concatenate stderr and stdout

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
LEADTIME=$(echo $TASK_LINE | awk '{print $3}')

echo "Sub-request ${PBS_SUBREQNO}: Processing year=$YEAR, month=$MONTH, leadtime=${LEADTIME}h"

# Set number of OpenMP threads for CDO
export OMP_NUM_THREADS=4

# Load CDO module
module load cdo

dataDir=/hpc/rwork2/evalpp/data/ICON_EU_EPS
targetDomain=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/target_grid.txt

# Grid switch date (YYYYMMDDHH format)
switchDate=2022112300
intWeights_old=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolation_weights/remap_gennn_ICON_EU_EPS.nc
intWeights_new=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolation_weights/remap_gennn_rea.nc

output_dir=/hpc/uwork/extmfeik/data
tmpDir_mean=${output_dir}/ensmean
tmpDir_std=${output_dir}/ensstd
mkdir -p $tmpDir_mean
mkdir -p $tmpDir_std

echo "Computing ensemble means and standard deviations for leadtime ${LEADTIME}h..."

leadtime=$LEADTIME

# Find all unique dates for this specific year and month
dates=$(find ${dataDir}/${YEAR}/${MONTH}/*/00 -type f 2>/dev/null \
    -regextype posix-extended \
    -regex ".*_s_${leadtime}_.*\.grib" \
    -printf "%P\n" | \
    grep -oP '\d{10}' | sort -u)

if [ -z "$dates" ]; then
    echo "No data found for ${YEAR}-${MONTH}, leadtime ${leadtime}h"
    exit 0
fi

# For each date, compute ensemble mean and std
for date in $dates; do

    # Check if this date/leadtime combination has already been computed
    if [ -f "${tmpDir_mean}/ensmean_${date}_${leadtime}.nc" ] && \
       [ -f "${tmpDir_std}/ensstd_${date}_${leadtime}.nc" ]; then
        echo "Skipping ${date} (leadtime ${leadtime}h) - already computed"
        continue
    fi

    # Extract date components (date format: YYYYMMDDHH)
    year=${date:0:4}
    month=${date:4:2}
    day=${date:6:2}
    hour=${date:8:2}

    # Skip dates at or after the grid switch date
    if [ $date -ge $switchDate ]; then
        echo "Skipping ${date} (leadtime ${leadtime}h) - at or after switch date ${switchDate}"
        continue
    fi
    intWeights=$intWeights_old

    # Construct the specific directory path
    dateDir=${dataDir}/${year}/${month}/${day}/${hour}

    # Check if directory exists
    if [ ! -d "$dateDir" ]; then
        continue
    fi

    # Get all member files for this date and lead time (only in this directory!)
    memberFiles=$(find ${dateDir} -maxdepth 1 -type f -name "*_${date}_mem_*_s_${leadtime}_*.grib" | sort -V)

    if [ -z "$memberFiles" ]; then
        continue
    fi

    echo "Computing ensmean and ensstd for ${date} (leadtime ${leadtime}h)..."

    # Compute ensemble mean for this date/leadtime
    if [ ! -f "${tmpDir_mean}/ensmean_${date}_${leadtime}.nc" ]; then
        cdo -f nc4 -z zip_6 \
            remap,$targetDomain,$intWeights \
            -ensmean \
            $memberFiles \
            ${tmpDir_mean}/ensmean_${date}_${leadtime}.nc
    fi

    # Compute ensemble standard deviation for this date/leadtime
    if [ ! -f "${tmpDir_std}/ensstd_${date}_${leadtime}.nc" ]; then
        cdo -f nc4 -z zip_6 \
            remap,$targetDomain,$intWeights \
            -ensstd \
            $memberFiles \
            ${tmpDir_std}/ensstd_${date}_${leadtime}.nc
    fi
done

echo "Done processing ${YEAR}-${MONTH}, leadtime ${LEADTIME}h!"
