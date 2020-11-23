#!/bin/bash

# This script run inside singularity container.

yaml_config="$1"
echo "Get test result for ${yaml_config}"

PYTHON_ENV=/opt/conda/envs/python37/bin/python
SRC_ROOT=/src_external

train_model () {
    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/src/train_n_fold.py \
                  --yaml-config ${yaml_config} \
                  --run-train "False" \
                  --run-test "True" \
                  --run-grad-cam "False" \
                  --train-fold -1
    set +o xtrace
}

generate_grad_cam () {
    local idx_fold="$1"

    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/src/train_n_fold.py \
                  --yaml-config ${yaml_config} \
                  --run-train "False" \
                  --run-test "False" \
                  --run-grad-cam "True" \
                  --train-fold ${idx_fold}
    set +o xtrace
}


#get_average_cam () {
#    local cam_folder=/proj_root/average_cam
#    set -o xtrace
#    ${PYTHON_ENV} ${SRC_ROOT}/src/grad_cam_analysis2.py \
#        --yaml-config ${yaml_config} \
#        --cam-folder ${cam_folder}
#    set +o xtrace
#}

train_model

for idx_fold in 0 1 2 3 4
do
    generate_grad_cam ${idx_fold}
done

#get_average_cam