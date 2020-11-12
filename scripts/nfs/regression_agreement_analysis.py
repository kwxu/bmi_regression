import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp
from src.tools.utils import get_logger
import yaml
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt


logger = get_logger('Regression analysis')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_5_cam.yaml')
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    num_fold = config['fold_num']
    exp_dir = config['exp_dir']

    gt_list = []
    pred_list = []
    rmse_list = []
    for idx_fold in range(num_fold):
        pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
        pred = pred_result_df['pred'].to_numpy()
        label = pred_result_df['target'].to_numpy()

        rmse_val = sqrt(mean_squared_error(label, pred))
        rmse_list.append(rmse_val)
        gt_list.append(label)
        pred_list.append(pred)

    rmse_list = np.array(rmse_list)
    print(f'RMSE mean: {np.mean(rmse_list):.4f}, std: {np.std(rmse_list):.4f}')

    pred_list = np.concatenate(pred_list)
    gt_list = np.concatenate(gt_list)

    f, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.mean_diff_plot(pred_list, gt_list, ax=ax)

    out_png = osp.join(exp_dir, 'mean_diff.png')
    logger.info(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    main()