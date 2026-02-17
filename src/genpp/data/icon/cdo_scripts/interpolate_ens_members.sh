#!/bin/bash -l

#PBS -N icon_eu_eps_ens_members     # Job name (will be modified per month)
#PBS -S /bin/bash                    # set the executing shell
#PBS -q rc_big                       # queue name
#PBS -l cpunum_job=4                 # use 4 CPUs (for CDO OpenMP)
#PBS -l memsz_job=8gb                # total memory for job
#PBS -l vmemsz_job=8gb               # total virtual memory
#PBS -l elapstim_req=03:00:00        # max runtime: 3 hours (per month)
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/logs/icon_ens_members_${YEAR}${MONTH}.log
#PBS -j o                            # concatenate stderr and stdout

# This script saves all 40 ensemble members per date/leadtime into a single .nc file.
# Only the 2 target variables (T_2M, VMAX_10M) are selected and remapped to the target grid.
# Output files are named: ens_{date}_{leadtime}.nc

# Year and Month should be env vars
if [ -z "$YEAR" ] || [ -z "$MONTH" ]; then
    echo "ERROR: Year and month required as environment variables"
    echo "Usage: YEAR=YYYY MONTH=MM $0"
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
tmpDir_ens=${output_dir}/ens
mkdir -p $tmpDir_ens

echo "Saving ensemble members..."

# For each lead time
for leadtime in 24 48 72 96 120; do
    echo "Processing lead time ${leadtime}h..."

    # Find all unique dates for this specific year and month (only 00 initialization)
    dates=$(find ${dataDir}/${YEAR}/${MONTH}/*/00 -type f 2>/dev/null \
        -regextype posix-extended \
        -regex ".*_s_${leadtime}_.*\.grib" \
        -printf "%P\n" | \
        grep -oP '\d{10}' | sort -u)

    if [ -z "$dates" ]; then
        echo "No data found for ${YEAR}-${MONTH}, leadtime ${leadtime}h"
        continue
    fi

    # For each date, save all ensemble members
    for date in $dates; do

        # Check if this date/leadtime combination has already been computed
        if [ -f "${tmpDir_ens}/ens_${date}_${leadtime}.nc" ]; then
            echo "Skipping ${date} (leadtime ${leadtime}h) - already computed"
            continue
        fi

        # Extract date components (date format: YYYYMMDDHH)
        year=${date:0:4}
        month=${date:4:2}
        day=${date:6:2}
        hour=${date:8:2}

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

        echo "Saving ensemble members for ${date} (leadtime ${leadtime}h)..."

        # Process each member: select T_2M and VMAX_10M, remap to target grid
        tmpMemDir=$(mktemp -d)
        memIdx=0
        for memberFile in $memberFiles; do
            cdo -f nc4 -z zip_6 \
                remap,$targetDomain,$intWeights \
                -selname,T_2M,VMAX_10M \
                $memberFile \
                ${tmpMemDir}/mem_$(printf "%03d" $memIdx).nc
            memIdx=$((memIdx + 1))
        done

        # Concatenate all processed members into a single file
        cdo -O cat ${tmpMemDir}/mem_*.nc ${tmpDir_ens}/ens_${date}_${leadtime}.nc

        # Clean up temporary member files
        rm -rf $tmpMemDir
    done
done

echo "Done processing ${YEAR}-${MONTH}!"
