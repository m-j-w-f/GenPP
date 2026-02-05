#!/bin/bash
module load cdo
gridFile=/hpc/rhome/routfox/routfox/icon/grids/public/edzw/icon_grid_0028_R02B07_N02.nc # Gridfile for the ICON_EPS
targetDomain=target_grid.txt
cdo gennn,$targetDomain $gridFile interpolation_weights/remap_gennn_ICON_EU_EPS.nc