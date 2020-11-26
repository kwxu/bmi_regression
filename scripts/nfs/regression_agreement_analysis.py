import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp
from src.tools.utils import get_logger, read_file_contents_list
import yaml
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tools.utils import mkdir_p


logger = get_logger('Regression analysis')


lower_bound = 12
upper_bound = 50


def mean_diff_plot(pred_list, gt_list, rmse_list, out_png):
    f, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.mean_diff_plot(pred_list, gt_list, ax=ax)
    ax.set_title(f'RMSE: {np.mean(rmse_list):.4f}, R2: {r2_score(gt_list, pred_list):.4f}')

    logger.info(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def scatter_plot(pred_list, gt_list, rmse_list, out_png):
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gt_list, pred_list, alpha=0.7)
    ax.set_xlabel(f'Ground-Truth BMI')
    ax.set_ylabel(f'Predicted BMI')
    ax.set_title(f'RMSE: {np.mean(rmse_list):.4f}, R2: {r2_score(gt_list, pred_list):.4f}')

    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle='--', alpha=0.7, c='r')

    logger.info(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def get_data(exp_dir, num_fold):
    gt_list = []
    pred_list = []
    rmse_list = []
    file_name_list = []

    mean_diff_list = []

    fold_output_root = os.path.join(exp_dir, 'pred_plot')
    mkdir_p(fold_output_root)
    for idx_fold in range(num_fold):
        pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
        pred = pred_result_df['pred'].to_numpy()
        label = pred_result_df['target'].to_numpy()
        file = pred_result_df['file_name'].to_list()

        file_name_list += file

        rmse_val = sqrt(mean_squared_error(label, pred))

        out_fold_mean_diff_png = os.path.join(fold_output_root, f'mean_diff_{idx_fold}.png')
        out_fold_scatter_png = os.path.join(fold_output_root, f'scatter_{idx_fold}.png')

        mean_diff_plot(pred, label, np.array([rmse_val]), out_fold_mean_diff_png)
        scatter_plot(pred, label, np.array([rmse_val]), out_fold_scatter_png)

        mean_diff = np.mean(pred - label)
        mean_diff_list.append(mean_diff)
        print(f'Mean diff: {mean_diff:.3f}')

        rmse_list.append(rmse_val)
        gt_list.append(label)
        pred_list.append(pred)

    rmse_list = np.array(rmse_list)
    # print(f'RMSE mean: {np.mean(rmse_list):.4f}, std: {np.std(rmse_list):.4f}')

    pred_list = np.concatenate(pred_list)
    gt_list = np.concatenate(gt_list)

    mean_diff_png = osp.join(exp_dir, 'mean_diff.png')
    scatter_png = osp.join(exp_dir, 'pred_scatter.png')
    mean_diff_plot(pred_list, gt_list, rmse_list, mean_diff_png)
    scatter_plot(pred_list, gt_list, rmse_list, scatter_png)

    print(f'mean diff: mean {np.mean(np.array(mean_diff_list))}, std {np.std(np.array(mean_diff_list))}')

    # output the diff
    data_dict = {
        'file_name': file_name_list,
        'pred': pred_list,
        'label': gt_list,
        'diff': pred_list - gt_list,
        'diff_abs': np.abs(pred_list - gt_list)
    }

    data_df = pd.DataFrame(data_dict)

    out_csv = os.path.join(exp_dir, f'pred_total.csv')
    print(f'Save to {out_csv}')
    data_df.to_csv(out_csv, index=False)


def analyze_top_error_cases(exp_dir):
    out_csv = os.path.join(exp_dir, f'pred_total.csv')
    print(f'Load csv {out_csv}')

    data_df = pd.read_csv(out_csv)

    data_df.sort_values(by=['diff_abs'], ascending=False)

    post_sort_diff_list = data_df['diff'].to_numpy()

    for idx_top in range(700):
        mean_diff = np.mean(post_sort_diff_list[:idx_top])
        if idx_top % 5 == 0:
            print(f'{idx_top}: {mean_diff:.4f}')


# yaml_config_name1 = 'simg_bmi_regression_3.6_nfs.yaml'
# yaml_config_name2 = 'simg_bmi_regression_0_1_nfs.yaml'
# yaml_config_name3 = 'simg_bmi_regression_0_2.yaml'
# yaml_config_name = 'simg_bmi_regression_0_3_nfs.yaml'
# yaml_config_name = 'simg_bmi_regression_3.6.3_nfs.yaml'
yaml_config_name = 'simg_bmi_regression_3.6.4_nfs.yaml'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default=yaml_config_name)
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    num_fold = config['fold_num']
    exp_dir = config['exp_dir']

    get_data(exp_dir, num_fold)
    analyze_top_error_cases(exp_dir)






if __name__ == '__main__':
    main()