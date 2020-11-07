#!/bin/bash

yaml_config="$1"

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
log_root=${PROJ_ROOT}/../run_log/${yaml_config}
mkdir -p ${log_root}

run_test () {
    local log_out=${log_root}/test.out
    local log_err=${log_root}/test.err

    set -o xtrace
    sbatch --error=${log_err} --output=${log_out} ${PROJ_ROOT}/sbatch.slurm train_model_run_test.sh ${yaml_config}
    set +o xtrace
}

run_test
