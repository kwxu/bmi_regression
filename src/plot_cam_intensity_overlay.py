import os
import numpy as np
from tools.utils import get_logger
import argparse
import yaml
import pandas as pd
import nibabel as nib
from tools.data_io import ScanWrapper
from tools.utils import mkdir_p
from src.tools.plot import ClipPlotSeriesWithBack


logger = get_logger('CAM intensity plot')


atlas_intensity_folder = '/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/valid_region/s6.1_int'


def main():
    yaml_config = 'simg_bmi_regression_3.6.3_nfs.yaml'

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    num_fold = config['fold_num']
    exp_dir = config['exp_dir']
    layer_flag = config['gcam_target_layer']

    gcam_overlay_out_folder = os.path.join(exp_dir, 'gcam_overlay')
    gcam_link_out_folder = os.path.join(exp_dir, 'gcam_link')
    mkdir_p(gcam_overlay_out_folder)
    mkdir_p(gcam_link_out_folder)

    # for idx_fold in range(num_fold):
    #     pred_result_csv = os.path.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
    #     cam_folder = os.path.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.{layer_flag}')
    #     pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
    #     file_list = pred_result_df['file_name']
    #
    #     for file_name in file_list:
    #         atlas_intensity_path = os.path.join(atlas_intensity_folder, file_name)
    #         cam_path = os.path.join(cam_folder, file_name)
    #
    #         plot_obj = ClipPlotSeriesWithBack(
    #             cam_path,
    #             None,
    #             atlas_intensity_path,
    #             10, 35, 15,
    #             3,
    #             0, 1,
    #             -1000, 600,
    #             None
    #         )
    #
    #         plot_obj.clip_plot(gcam_overlay_out_folder)

    for idx_fold in range(num_fold):
        pred_result_csv = os.path.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        cam_folder = os.path.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.{layer_flag}')
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
        file_list = pred_result_df['file_name']

        for file_name in file_list:
            cam_path = os.path.join(cam_folder, file_name)
            link_path = os.path.join(gcam_link_out_folder, file_name)
            cmd_str = f'cp -f {cam_path} {link_path}'
            # cmd_str = f'ln -s {cam_path} {link_path}'
            logger.info(cmd_str)
            os.system(cmd_str)


if __name__ == '__main__':
    main()