#!/bin/bash

config_file=$1

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python
SRC_ROOT=${PROJ_ROOT}/../..

for idx in 0 1 2 3 5
do
    yaml_config="simg_bmi_regression_${idx}_cam.yaml"
    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/src/get_training_curve.py \
              --yaml-config ${yaml_config}
    set +o xtrace
done
