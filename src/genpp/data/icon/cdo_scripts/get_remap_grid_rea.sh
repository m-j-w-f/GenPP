#!/bin/bash
module load cdo
gridFile=/hpc/rhome/routfox/routfox/icon/grids/public/edzw/icon_grid_0037_R03B07_N02.nc # Gridfile for the reanalysis
targetDomain=target_grid.txt
cdo gennn,$targetDomain $gridFile interpolation_weights/remap_gennn_rea.nc