#!/bin/bash

PROJ_ROOT=/nfs/masi/xuk9/SPORE/CAC_class

PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/Thorax_non_rigid_combine

IN_MASK_FOLDER=/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/valid_region/s3_overlapped
OUT_AVERAGE_PATH=/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/valid_region/s3_overlapped_average.nii.gz

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/tools/paral_average_only.py \
    --in-folder ${IN_MASK_FOLDER} \
    --out-average-path ${OUT_AVERAGE_PATH}
set +o xtrace

