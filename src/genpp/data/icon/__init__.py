from pathlib import Path

from genpp import BASE_DIR

ICON_EU_REA_PATH: Path = Path("/hpc/rwork2/evalpp/data/rea_grid_0037_R03B07")
ICON_EU_ENS_PATH: Path = Path("/hpc/rwork2/evalpp/data/ICON_EU_EPS")

DATA_DIR = BASE_DIR / "data" / "icon" / "data"

LEVELS_TO_FLATTEN = [
    "plev",
    "plev_2",
    "plev_3",
    "plev_4",
    "plev_5",
    "depth",
    "lev",
    "height",
    "height_2",
]
VARS_TO_DROP = ["rotated_pole", "plev_bnds", "plev_2_bnds", "plev_3_bnds", "depth_bnds"]
AXIS_ORDER = ["feature", "x", "y"]

VARS_GRID_28 = [
    "ALB_RAD",
    "ASOB_S",
    "ASOB_T",
    "ATHB_S",
    "ATHB_T",
    "CLCH+plev_0.0",
    "CLCL+plev_2_80000.0",
    "CLCM+plev_3_40000.0",
    "CLCT",
    "HBAS_CON",
    "HTOP_CON",
    "PMSL",
    "RAIN_CON",
    "RAIN_GSP",
    "SNOW_CON",
    "SNOW_GSP",
    "SOBS_RAD",
    "TD_2M+height_2.0",
    "THBS_RAD",
    "TMAX_2M+height_2.0",
    "TMIN_2M+height_2.0",
    "TOT_PREC",
    "TQC",
    "TQI",
    "TQV",
    "T_2M+height_2.0",
    "T_G",
    "U_10M+height_2_10.0",
    "VMAX_10M+height_2_10.0",
    "V_10M+height_2_10.0",
    "W_SNOW",
    "W_SO+depth_0.0",
    "W_SO+depth_0.01",
    "W_SO+depth_0.03",
    "W_SO+depth_0.09",
    "W_SO+depth_0.27",
    "W_SO+depth_0.81",
    "W_SO+depth_2.43",
    "W_SO+depth_7.29",
    "Z0",
]
