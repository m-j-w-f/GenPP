# Parallel Tensor Processing Scripts

This directory contains scripts for processing ICON forecast and reanalysis data into tensors in parallel using NQSV job submissions.

## Files

- **`process_tensors.py`**: Standalone Python script that processes NetCDF files into PyTorch tensors
- **`submit_fc_tensors.sh`**: NQSV job submission script for forecast (FC) tensors
- **`submit_rea_tensors.sh`**: NQSV job submission script for reanalysis (REA) tensors
- **`launch_tensor_jobs.sh`**: Launcher script that submits jobs for all months in a date range

## Overview

The `_get_fc_tensors` and `_get_rea_tensors` functions from `dataset.py` have been extracted into a standalone script to enable parallel processing. These functions take a significant amount of time to run, so processing them in parallel by month significantly reduces the total runtime.

This implementation follows the same pattern as the existing `launch_interpolate_ens_jobs.sh` and `interpolate_ens.sh` scripts in the `cdo_scripts` directory.

## Usage

### 1. Prepare Log Directory

Before submitting jobs, create a directory for log files:

```bash
mkdir -p logs
```

### 2. Modify Date Range (if needed)

Edit the year range in `launch_tensor_jobs.sh` to match your data range:

```bash
for year in {2018..2024}; do
    for month in {01..12}; do
        # Skip future months for 2024 (adjust as needed)
        if [ $year -eq 2024 ] && [ $month -gt 09 ]; then
            continue
        fi
        ...
    done
done
```

### 3. Submit Jobs

**Important:** Run the launcher script from the `src/genpp/data/icon` directory:

```bash
# Navigate to the icon data directory
cd src/genpp/data/icon

# Submit all jobs for the configured date range
bash launch_tensor_jobs.sh
```

This will submit separate jobs for each month, with both FC and REA tensors processed in parallel.

### Alternative: Manual Job Submission

You can also submit individual jobs manually:

```bash
# Navigate to the icon data directory
cd src/genpp/data/icon

# Submit a single FC tensor job for January 2019
qsub -N fc_tensor_201901 -o logs/fc_tensor_201901.log -v YEAR=2019,MONTH=01 submit_fc_tensors.sh

# Submit a single REA tensor job for January 2019
qsub -N rea_tensor_201901 -o logs/rea_tensor_201901.log -v YEAR=2019,MONTH=01 submit_rea_tensors.sh
```

### 4. Monitor Jobs

Check job status:

```bash
qstat -u $USER
```

View logs:

```bash
# View specific month logs
tail -f logs/fc_tensor_201901.log
tail -f logs/rea_tensor_201901.log
```

## Manual Testing

You can also run the script manually for testing:

```bash
# Test processing FC tensors for January 2019
JOB_TYPE=fc YEAR=2019 MONTH=01 pixi run -e nb python process_tensors.py

# Test processing REA tensors for January 2019
JOB_TYPE=rea YEAR=2019 MONTH=01 pixi run -e nb python process_tensors.py
```

## Variable Selection

The script uses the following variable selections as specified:

**X variables (input features):**
```python
['ALB_RAD', 'ASOB_S', 'ASOB_T', 'ATHB_S', 'ATHB_T', 'CLCH+plev_0.0',
 'CLCL+plev_2_80000.0', 'CLCM+plev_3_40000.0', 'CLCT', 'HBAS_CON',
 'HTOP_CON', 'PMSL', 'RAIN_CON', 'RAIN_GSP', 'SNOW_CON', 'SNOW_GSP',
 'SOBS_RAD', 'TD_2M+height_2.0', 'THBS_RAD', 'TMAX_2M+height_2.0',
 'TMIN_2M+height_2.0', 'TOT_PREC', 'TQC', 'TQI', 'TQV',
 'T_2M+height_2.0', 'T_G', 'U_10M+height_2_10.0',
 'VMAX_10M+height_2_10.0', 'V_10M+height_2_10.0', 'W_SNOW',
 'W_SO+depth_0.0', 'W_SO+depth_0.01', 'W_SO+depth_0.03',
 'W_SO+depth_0.09', 'W_SO+depth_0.27', 'W_SO+depth_0.81',
 'W_SO+depth_2.43', 'W_SO+depth_7.29', 'Z0']
```

**Y variables (prediction targets):**
```python
['T_2M+height_2.0', 'VMAX_10M+height_2_10.0']
```

## Output

Processed tensors are saved to:
- FC tensors: `data/icon/data/tensors/fc/`
- Meta tensors: `data/icon/data/tensors/meta/`
- REA tensors: `data/icon/data/tensors/rea/`

The scripts automatically skip files that have already been processed, so you can safely re-run them if a job fails.

## Resource Requirements

Current settings in the submission scripts:
- **Time limit**: 12 hours per task
- **CPUs**: 4 cores per task
- **Memory**: 32GB per task

Adjust these in the `#PBS` directives if needed based on your data size and cluster resources.

## Troubleshooting

1. **Jobs receive SIGTERM immediately**: 
   - The scripts now automatically add pixi to PATH from common installation locations (`~/.pixi/bin` or `~/.local/bin`)
   - If pixi is installed elsewhere, you may need to modify the PATH setup in the submission scripts
   - Check the job logs for "ERROR: pixi command not found in PATH"
   
2. **Jobs fail immediately**: Check that the data directory exists and contains the expected ensmean/ensstd/rea subdirectories

3. **Out of memory**: Increase `memsz_job` in the submission scripts

4. **Timeout**: Increase `elapstim_req` in the submission scripts

5. **Missing pixi environment**: 
   - Ensure pixi is installed (typically in `~/.pixi/bin/pixi` or `~/.local/bin/pixi`)
   - Ensure the `nb` environment is set up with: `pixi install`
   - Test manually: `JOB_TYPE=fc YEAR=2019 MONTH=01 pixi run -e nb python process_tensors.py`

## Notes

- The script automatically creates output directories if they don't exist
- Already-processed files are skipped to allow resuming failed jobs
- Each month's data is processed independently, enabling full parallelization
- The scheduler (qsub) uses the NQSV system as documented in `qsub.txt`
