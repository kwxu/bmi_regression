#!/bin/bash

config_file=$1

PROJ_ROOT="/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male"

PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/Thorax_non_rigid_combine

get_cam() {
    local idx="$1"
    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/tools/paral_average_only.py \
              --in-folder ${PROJ_ROOT}/output/non_rigid/interp/cam_heatmap \
              --out-average-path ${PROJ_ROOT}/output/non_rigid/interp/cam_heatmap_average.nii.gz \
              --num-process 10
    set +o xtrace
}

get_cam