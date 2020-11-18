import numpy as np
import argparse
import os.path as osp
import os
from tools.utils import get_logger
import yaml
import pandas as pd
from sklearn.metrics import r2_score


logger = get_logger('Get R Square')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_18_nfs.yaml')
    # parser.add_argument('--idx-fold', type=int, default=4)
    args = parser.parse_args()

    r2_list = []
    for idx_fold in range(5):
        SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/..'
        yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
        logger.info(f'Read yaml file {yaml_config}')
        f = open(yaml_config, 'r').read()
        config = yaml.safe_load(f)

        exp_dir = config['exp_dir']

        pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)

        pred = pred_result_df['pred'].to_numpy()
        label = pred_result_df['target'].to_numpy()

        r_square = r2_score(pred, label)
        print(f'Fold {idx_fold}, R2 = {r_square:.3f}')

        r2_list.append(r_square)

    r2_list = np.array(r2_list)
    print(f'R2 mean: {np.mean(r2_list):.3f}, std: {np.std(r2_list)}')


if __name__ == '__main__':
    main()
