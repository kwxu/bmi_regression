#!/bin/bash

yaml_config="$1"
fold_idx="$2"
echo "Run single fold (${fold_idx} / 5) with ${yaml_config}"

PYTHON_ENV=/opt/conda/envs/python37/bin/python
SRC_ROOT=/src_external

train_model () {
    set -o xtrace
    ${PYTHON_ENV} ${SRC_ROOT}/src/train_n_fold.py \
                  --yaml-config ${yaml_config} \
                  --run-train "True" \
                  --run-test "False" \
                  --run-grad-cam "False" \
                  --train-fold ${fold_idx}
    set +o xtrace
}

train_model
