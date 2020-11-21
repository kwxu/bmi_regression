import pandas as pd
import argparse
import os
from src.tools.utils import get_logger
import yaml


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


yaml_config_name1 = 'simg_bmi_regression_0_1_nfs.yaml'
yaml_config_name2 = 'simg_bmi_regression_3.6_nfs.yaml'
out_csv = '/nfs/masi/xuk9/SPORE/CAC_class/output/simg_bmi_regression_3.6/diff_native.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config1', type=str, default=yaml_config_name1)
    parser.add_argument('--yaml-config2', type=str, default=yaml_config_name2)
    args = parser.parse_args()

    test_1_df = get_the_total_perf_data_dict(args.yaml_config1)
    test_2_df = get_the_total_perf_data_dict(args.yaml_config2)

    data_total_dict = {
        'file_name': test_1_df['file_name'].to_list(),
        'label': test_1_df['label'],
        'pred1': test_1_df['pred'].to_numpy(),
        'pred2': test_2_df['pred'].to_numpy(),
        'diff1': test_1_df['diff'].to_numpy(),
        'diff2': test_2_df['diff'].to_numpy(),
        'diff1_abs': test_1_df['diff_abs'].to_numpy(),
        'diff2_abs': test_2_df['diff_abs'].to_numpy(),
        'diff_1_2': test_1_df['pred'] - test_2_df['pred'],
        'diff_abs_1_2': test_1_df['diff_abs'] - test_2_df['diff_abs']
    }

    data_total_df = pd.DataFrame(data_total_dict)
    print(f'Save to {out_csv}')
    data_total_df.to_csv(out_csv, float_format='%.3f', index=False)


if __name__ == '__main__':
    main()