#!/bin/bash -l

#PBS -N icon_eu_eps_ens           # Job name (will be modified per month)
#PBS -S /bin/bash                  # set the executing shell
#PBS -q rc_big                     # queue name
#PBS -l cpunum_job=4              # use 4 CPUs (for CDO OpenMP)
#PBS -l memsz_job=8gb            # total memory for job
#PBS -l vmemsz_job=8gb           # total virtual memory
#PBS -l elapstim_req=03:00:00     # max runtime:  2 hours (per month)
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/icon_ens_${YEAR}${MONTH}.log
#PBS -j o                          # concatenate stderr and stdout

# Year and Month should be env vars
if [ -z "$YEAR" ] || [ -z "$MONTH" ]; then
    echo "ERROR: Year and month required as arguments"
    echo "Usage: $0 YYYY MM"
    exit 1
fi

echo "Processing year: $YEAR, month: $MONTH"

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

echo "Computing ensemble means and standard deviations..."

# For each lead time
for leadtime in 24 48 72 96 120; do
    echo "Processing lead time ${leadtime}h..."

    # Find all unique dates for this specific year and month
    dates=$(find ${dataDir}/${YEAR}/${MONTH}/*/00 -type f 2>/dev/null \
        -regextype posix-extended \
        -regex ".*_s_${leadtime}_.*\.grib" \
        -printf "%P\n" | \
        grep -oP '\d{10}' | sort -u)

    if [ -z "$dates" ]; then
        echo "No data found for ${YEAR}-${MONTH}, leadtime ${leadtime}h"
        continue
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
        hour=${date: 8:2}

        # Select appropriate interpolation weights based on date
        if [ $date -lt $switchDate ]; then
            intWeights=$intWeights_old
        else
            intWeights=$intWeights_new
        fi

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
        if [ !  -f "${tmpDir_mean}/ensmean_${date}_${leadtime}.nc" ]; then
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
done

echo "Done processing ${YEAR}-${MONTH}!"

# # Merging all ensemble means...
# echo "Merging all ensemble means..."
# sortedFiles=$(ls ${tmpDir_mean}/ensmean_*.nc | sort -V)
# cdo -f nc4 -z zip_6 cat $sortedFiles ${output_dir}/icon_eu_eps_ensmean. nc
#
# # Merging all ensemble standard deviations...
# echo "Merging all ensemble standard deviations..."
# sortedFiles=$(ls ${tmpDir_std}/ensstd_*.nc | sort -V)
# cdo -f nc4 -z zip_6 cat $sortedFiles ${output_dir}/icon_eu_eps_ensstd.nc
#
# echo "Cleaning up..."
# # rm -rf $tmpDir_mean
# # rm -rf $tmpDir_std
#
# echo "Done!   Outputs:"
# echo "  Mean: ${output_dir}/icon_eu_eps_ensmean. nc"
# echo "  Std:   ${output_dir}/icon_eu_eps_ensstd. nc"
# cdo sinfov ${output_dir}/icon_eu_eps_ensmean.nc
# cdo sinfov ${output_dir}/icon_eu_eps_ensstd.nc
