import argparse
import os
import subprocess
import re
import errno


exp_path = '/scratch/xuk9/nifti/SPORE/CAC_class'
proj_path = os.path.dirname(os.path.realpath(__file__))
slurm_path = os.path.join(proj_path, '../simg/slurm/sbatch.slurm')


def mkdir_p(path):
    print(f'mkdir: {path}')
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def run_single_fold(fold_idx, log_root, yaml_config):
    log_out = os.path.join(log_root, f'fold_{fold_idx}.out')
    log_err = os.path.join(log_root, f'fold_{fold_idx}.err')

    cmd_str = f'sbatch --error={log_err} --output={log_out} {slurm_path} train_model_single_fold.sh {yaml_config} {fold_idx}'
    print(cmd_str)
    cmd_status = subprocess.getstatusoutput(cmd_str)
    print(cmd_status)
    match_list = re.match(r"Submitted batch job (?P<job_id>\d+)", cmd_status[1])
    job_id = match_list.group('job_id')
    return job_id


def run_perf_test(log_root, yaml_config, dep_job_id_list):
    log_out = os.path.join(log_root, f'test.out')
    log_err = os.path.join(log_root, f'test.err')

    dep_str = '--dependency=afterok'
    for job_id in dep_job_id_list:
        dep_str += f':{job_id}'

    cmd_str = f'sbatch {dep_str} --error={log_err} --output={log_out} {slurm_path} train_model_run_test.sh {yaml_config}'
    print(cmd_str)
    cmd_status = subprocess.getstatusoutput(cmd_str)
    print(cmd_status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='experiments_n_fold_3.yaml')
    args = parser.parse_args()

    log_root = os.path.join(exp_path, f'run_log/{args.yaml_config}')
    mkdir_p(log_root)

    job_id_array = []
    for fold_idx in range(5):
        job_id = run_single_fold(fold_idx, log_root, args.yaml_config)
        job_id_array.append(job_id)

    run_perf_test(log_root, args.yaml_config, job_id_array)


if __name__ == '__main__':
    main()
