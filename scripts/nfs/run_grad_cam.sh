#!/bin/bash

config_file=$1

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python
SRC_ROOT=${PROJ_ROOT}/../..

get_cam() {
    local idx="$1"
    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/src/train_n_fold.py \
              --yaml-config ${config_file} \
              --run-train "False" \
              --run-test "False" \
              --run-grad-cam "True" \
              --train-fold ${idx}
    set +o xtrace
}

get_average() {
    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/src/grad_cam_analysis2.py \
        --yaml-config ${config_file}
    set +o xtrace
}

#for idx in {0..4}
#do
#    get_cam ${idx}
#done

get_average