import os.path as osp
import os
import numpy as np
from tools.utils import get_logger
import argparse
import yaml
import pandas as pd
from tools.clinical import ClinicalDataReaderSPORE
import nibabel as nib
from tools.data_io import ScanWrapper
from sklearn.metrics import roc_curve
from tools.utils import mkdir_p
from tools.plot import plot_training_curve
import argparse


logger = get_logger('grad_cam_analysis')


# yaml_config_file = 'simg_bmi_regression_0_cam.yaml'
fold_idx = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_0_cam.yaml')
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    exp_dir = config['exp_dir']

    train_loss_file = os.path.join(exp_dir, f'fold_{fold_idx}/training/train_epoch_loss.txt')
    train_loss = np.loadtxt(train_loss_file)[:, 1]

    valid_loss_file = os.path.join(exp_dir, f'fold_{fold_idx}/validation/validation_epoch_loss.txt')
    valid_loss = np.loadtxt(valid_loss_file)[:, 1]

    if len(valid_loss) > len(train_loss):
        train_loss = train_loss[-len(valid_loss):]

    out_png = os.path.join(exp_dir, f'../../training_curve/{args.yaml_config}_fold_{fold_idx}.png')

    data_dict = {
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'start_epoch': 2,
        'end_epoch': 150,
        'loss_str': "Mean Squared Error"
    }

    plot_training_curve(data_dict, out_png)


if __name__ == '__main__':
    main()