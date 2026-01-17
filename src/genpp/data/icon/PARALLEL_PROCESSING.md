# Parallel Tensor Processing Scripts

This directory contains scripts for processing ICON forecast and reanalysis data into tensors in parallel using array jobs.

## Files

- **`process_tensors.py`**: Standalone Python script that processes NetCDF files into PyTorch tensors
- **`submit_fc_tensors.sh`**: qsub job submission script for forecast (FC) tensors
- **`submit_rea_tensors.sh`**: qsub job submission script for reanalysis (REA) tensors

## Overview

The `_get_fc_tensors` and `_get_rea_tensors` functions from `dataset.py` have been extracted into a standalone script to enable parallel processing. These functions take a significant amount of time to run, so processing them in parallel by month significantly reduces the total runtime.

## Usage

### 1. Prepare Log Directory

Before submitting jobs, create a directory for log files:

```bash
mkdir -p logs
```

### 2. Modify Month Lists (if needed)

Edit the `MONTHS` array in both submission scripts to match your data range:

```bash
declare -a MONTHS=(
    "2019-01" "2019-02" "2019-03" "2019-04" "2019-05" "2019-06"
    "2019-07" "2019-08" "2019-09" "2019-10" "2019-11" "2019-12"
    "2020-01" "2020-02" "2020-03" "2020-04" "2020-05" "2020-06"
    "2020-07" "2020-08" "2020-09" "2020-10" "2020-11" "2020-12"
)
```

Also update the `-t` parameter in the `#PBS` directives to match the number of months (e.g., `-t 1-24` for 24 months).

### 3. Submit Jobs

Submit the array jobs using qsub:

```bash
# Submit FC tensor processing (forecast data)
qsub submit_fc_tensors.sh

# Submit REA tensor processing (reanalysis data)
qsub submit_rea_tensors.sh
```

Each job will spawn multiple parallel tasks (one per month), and each task will process all days within that month.

### 4. Monitor Jobs

Check job status:

```bash
qstat
```

View logs (while jobs are running or after completion):

```bash
# View FC logs
tail -f logs/fc_tensor_1.out
tail -f logs/fc_tensor_2.out

# View REA logs
tail -f logs/rea_tensor_1.out
tail -f logs/rea_tensor_2.out
```

## Manual Testing

You can also run the script manually for testing:

```bash
# Test processing FC tensors for January 2019
JOB_TYPE=fc YEAR_MONTH=2019-01 pixi run -e nb python process_tensors.py

# Test processing REA tensors for January 2019
JOB_TYPE=rea YEAR_MONTH=2019-01 pixi run -e nb python process_tensors.py
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

1. **Jobs fail immediately**: Check that the data directory exists and contains the expected ensmean/ensstd/rea subdirectories
2. **Out of memory**: Increase `memsz_job` in the submission scripts
3. **Timeout**: Increase `elapstim_req` in the submission scripts
4. **Missing pixi environment**: Ensure pixi is installed and the `nb` environment is set up

## Notes

- The script automatically creates output directories if they don't exist
- Already-processed files are skipped to allow resuming failed jobs
- Each month's data is processed independently, enabling full parallelization
- The scheduler (qsub) uses the NQSV system as documented in `qsub.txt`
