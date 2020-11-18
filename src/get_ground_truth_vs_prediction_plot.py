import numpy as np
import argparse
import os.path as osp
import os
from tools.utils import get_logger
import yaml
import pandas as pd
from sklearn.metrics import r2_score
from tools.utils import mkdir_p
from tools.plot import plot_prediction_scatter


logger = get_logger('Plot Ground Truth vs. Prediction scatter')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_18_nfs.yaml')
    # parser.add_argument('--idx-fold', type=int, default=4)
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    scatter_plot_dir = os.path.join('/nfs/masi/xuk9/SPORE/CAC_class/prediction_plots', args.yaml_config)
    mkdir_p(scatter_plot_dir)

    for idx_fold in range(5):
        exp_dir = config['exp_dir']

        pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)

        out_png = os.path.join(scatter_plot_dir, f'fold_{idx_fold}.png')

        plot_prediction_scatter(pred_result_df, out_png)




if __name__ == '__main__':
    main()
