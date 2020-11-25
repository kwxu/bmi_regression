import pandas as pd
import argparse
import os
from src.tools.utils import get_logger, read_file_contents_list
import yaml
from src.tools.utils import mkdir_p
import numpy as np


logger = get_logger('Regression analysis')


def get_the_total_perf_data_dict(yaml_config):
    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    exp_dir = config['exp_dir']

    out_csv = os.path.join(exp_dir, f'pred_total.csv')
    print(f'Load csv {out_csv}')

    data_df = pd.read_csv(out_csv)

    return data_df


def get_statics_subgroup(file_list_txt, in_csv):
    # out_csv = os.path.join(exp_dir, 'pred_total.csv')
    print(f'Load csv {in_csv}')

    data_df = pd.read_csv(in_csv, index_col='file_name')

    study_file_list = read_file_contents_list(file_list_txt)
    num_study_case = len(study_file_list)
    print(f'# Study files: {num_study_case}')
    study_data_df = data_df.loc[study_file_list]

    study_label_list = study_data_df['label'].to_numpy()
    study_pred1_list = study_data_df['pred1'].to_numpy()
    study_pred2_list = study_data_df['pred2'].to_numpy()

    diff1 = study_pred1_list - study_label_list
    diff2 = study_pred2_list - study_label_list

    diff1_rmse = np.sqrt(np.sum(np.abs(diff1)) / num_study_case)
    diff1_mean = np.mean(diff1)
    diff2_rmse = np.sqrt(np.sum(np.abs(diff2)) / num_study_case)
    diff2_mean = np.mean(diff2)

    print(f'Diff1 RMSE: {diff1_rmse:.3f}')
    print(f'Diff1 Mean: {diff1_mean:.3f}')
    print(f'Diff2 RMSE: {diff2_rmse:.3f}')
    print(f'Diff2 Mean: {diff2_mean:.3f}')

# yaml_config_name1 = 'simg_bmi_regression_0_1_nfs.yaml'
# yaml_config_name2 = 'simg_bmi_regression_3.6_nfs.yaml'
yaml_config_name1 = 'simg_bmi_regression_0_3_nfs.yaml'
yaml_config_name2 = 'simg_bmi_regression_3.6.3_nfs.yaml'
out_folder = f'/nfs/masi/xuk9/SPORE/CAC_class/diff_analysis/{yaml_config_name1}'
mkdir_p(out_folder)
# out_csv = f'/nfs/masi/xuk9/SPORE/CAC_class/diff_analysis/diff_native.csv'
out_csv = os.path.join(out_folder, f'{yaml_config_name2}.csv')

file_list_round_fov_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/fov/round_fov'
file_list_normal_fov_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/fov/normal_fov'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config1', type=str, default=yaml_config_name1)
    parser.add_argument('--yaml-config2', type=str, default=yaml_config_name2)
    args = parser.parse_args()

    # test_1_df = get_the_total_perf_data_dict(args.yaml_config1)
    # test_2_df = get_the_total_perf_data_dict(args.yaml_config2)
    #
    # data_total_dict = {
    #     'file_name': test_1_df['file_name'].to_list(),
    #     'label': test_1_df['label'],
    #     'pred1': test_1_df['pred'].to_numpy(),
    #     'pred2': test_2_df['pred'].to_numpy(),
    #     'diff1': test_1_df['diff'].to_numpy(),
    #     'diff2': test_2_df['diff'].to_numpy(),
    #     'diff1_abs': test_1_df['diff_abs'].to_numpy(),
    #     'diff2_abs': test_2_df['diff_abs'].to_numpy(),
    #     'diff_1_2': test_1_df['pred'] - test_2_df['pred'],
    #     'diff_abs_1_2': test_1_df['diff_abs'] - test_2_df['diff_abs']
    # }
    #
    # data_total_df = pd.DataFrame(data_total_dict)
    # print(f'Save to {out_csv}')
    # data_total_df.to_csv(out_csv, float_format='%.3f', index=False)

    get_statics_subgroup(file_list_round_fov_txt, out_csv)
    get_statics_subgroup(file_list_normal_fov_txt, out_csv)


if __name__ == '__main__':
    main()