#!/bin/bash
# This script is used to put all reanalysis files at 00:00 and 12:00 in one .nc file.
# We only select the variables T_2M and VMAX_10M
# To precompute the remap grid use get_remap_grid_rea.sh

module load cdo

# Paths
dataDir=/hpc/rwork2/evalpp/data/rea_grid_0037_R03B07
targetDomain=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/target_grid.txt
intWeights=/hpc/uhome/extmfeik/GenPP/src/genpp/data/icon/cdo_scripts/interpolation_weights/remap_gennn_rea.nc

# Output
output_dir=/hpc/uwork/extmfeik/data
fnameNC=${output_dir}/rea.nc

# Temporary directory for yearly files
tmpDir=${output_dir}/tmp_yearly
mkdir -p $tmpDir

echo "Stage 1: Processing each year separately..."
for year in $(seq 2018 2024); do
    echo "Processing year $year..."

    # Get only files ending in 00 and 12 for this year
    gribFiles=$(find ${dataDir}/${year}/ -type f \( -name "*00.grib" -o -name "*12.grib" \) | sort)

    # Check if files were found
    if [ -z "$gribFiles" ]; then
        echo "No GRIB files found for $year, skipping..."
        continue
    fi

    fileCount=$(echo $gribFiles | wc -w)
    echo "Found $fileCount files for $year"

    # Process year
    cdo -f nc4 -z zip_6 \
        remap,$targetDomain,$intWeights \
        -selname,T_2M,VMAX_10M \
        -mergetime,names=union \
        $gribFiles \
        ${tmpDir}/rea_${year}.nc

    echo "Done:  ${tmpDir}/rea_${year}.nc"
done

echo ""
echo "Stage 2: Merging all years into single file..."
cdo -f nc4 -z zip_6 mergetime,names=union ${tmpDir}/rea_*.nc $fnameNC

echo ""
echo "Cleaning up temporary files..."
rm -rf $tmpDir

echo ""
echo "Done! Final output:  $fnameNC"
cdo sinfov $fnameNC