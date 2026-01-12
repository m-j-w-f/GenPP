#!/bin/bash -l
# Old file use interpolate2.sh with the launch job script
#PBS -N icon_eu_eps_ensmean           # Job name
#PBS -S /bin/bash                      # set the executing shell
#PBS -q rc_big                         # queue name
#PBS -l cpunum_job=4                   # use 4 CPUs (for CDO OpenMP)
#PBS -l memsz_job=32gb                # total memory for job
#PBS -l vmemsz_job=32gb               # total virtual memory
#PBS -l elapstim_req=4:00:00          # max runtime:  4 hours
#PBS -o /hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/icon_ensmean2.log               # logfile
#PBS -j o                              # concatenate stderr and stdout

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
tmpDir=${output_dir}/ensmean
mkdir -p $tmpDir

echo "Computing ensemble means..."

# For each lead time
for leadtime in 24 48 72 96 120; do
    echo "Processing lead time ${leadtime}h..."

    # Find all unique dates
    dates=$(find ${dataDir}/{2018..2024}/*/*/00 -type f \
        -regextype posix-extended \
        -regex ".*_s_${leadtime}_.*\.grib" \
        -printf "%P\n" | \
        grep -oP '\d{10}' | sort -u)

    # For each date, compute ensemble mean
    for date in $dates; do

        # Check if this date/leadtime combination has already been computed
        if [ -f "${tmpDir}/ensmean_${date}_${leadtime}.nc" ]; then
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

        echo "Computing ensmean for ${date} (leadtime ${leadtime}h)..."
        # Compute ensemble mean for this date/leadtime
        cdo -f nc4 -z zip_6 \
            remap,$targetDomain,$intWeights \
            -ensmean \
            $memberFiles \
            ${tmpDir}/ensmean_${date}_${leadtime}.nc
    done
done

#echo "Merging all ensemble means..."

# Sort files by date then leadtime
#sortedFiles=$(ls ${tmpDir}/ensmean_*.nc | sort -V)

# Use cat instead of mergetime (avoids opening all files at once)
#cdo -f nc4 -z zip_6 cat $sortedFiles ${output_dir}/icon_eu_eps_ensmean.nc

#echo "Cleaning up..."
# rm -rf $tmpDir

#echo "Done!  Output:  ${output_dir}/icon_eu_eps_ensmean.nc"
#cdo sinfov ${output_dir}/icon_eu_eps_ensmean.nc

echo "DONE!"