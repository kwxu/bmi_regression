import os
import numpy as np
from tools.utils import get_logger
import argparse
import yaml
import pandas as pd
import nibabel as nib
from tools.data_io import ScanWrapper, DataFolder
from tools.utils import mkdir_p
from tools.plot import ClipPlotSeriesWithBack, ClipPlotIntensityDeformationWall
from tools.paral import AbstractParallelRoutine


logger = get_logger('CAM intensity plot')


atlas_intensity_folder = '/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/valid_region/s6.1_int'
atlas_jacobian_folder = '/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/valid_region/s6.2_jac'


class ParaPlotClip(AbstractParallelRoutine):
    def __init__(
            self,
            in_int_folder_obj,
            in_jac_folder_obj,
            in_att_folder_obj,
            out_png_folder,
            num_process):
        super().__init__(in_int_folder_obj, num_process)
        self._in_jac_folder_obj = in_jac_folder_obj
        self._in_att_folder_obj = in_att_folder_obj
        self._out_png_folder = out_png_folder

    def _run_single_scan(self, idx):
        in_int_path = self._in_data_folder.get_file_path(idx)
        in_jac_path = self._in_jac_folder_obj.get_file_path(idx)
        in_att_path = self._in_att_folder_obj.get_file_path(idx)

        plot_obj = ClipPlotIntensityDeformationWall(
            in_int_path,
            in_jac_path,
            in_att_path,
            10, 35, 15,
            5,
            -1000, 600,
            -0.75, 0.75,
            0, 1
        )

        plot_obj.clip_plot(self._out_png_folder)


def main():
    # yaml_config = 'simg_bmi_regression_3.6.3_nfs.yaml'
    yaml_config = 'simg_bmi_regression_3.6.4_nfs.yaml'
    # yaml_config = 'simg_bmi_regression_3.6.6.1_nfs.yaml'

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

    pred_result_csv = os.path.join(exp_dir, f'pred_total.csv')
    file_list = pd.read_csv(pred_result_csv, index_col=False)['file_name']

    in_int_folder_obj = DataFolder(atlas_intensity_folder, file_list)
    in_jac_folder_obj = DataFolder(atlas_jacobian_folder, file_list)
    in_att_folder_obj = DataFolder(gcam_link_out_folder, file_list)

    paral_plot_obj = ParaPlotClip(
        in_int_folder_obj,
        in_jac_folder_obj,
        in_att_folder_obj,
        gcam_overlay_out_folder,
        10
    )

    paral_plot_obj.run_parallel()

    # for idx_fold in range(num_fold):
    #     pred_result_csv = os.path.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
    #     cam_folder = os.path.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.{layer_flag}')
    #     pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
    #     file_list = pred_result_df['file_name']
    #
    #     for file_name in file_list:
    #         cam_path = os.path.join(cam_folder, file_name)
    #         link_path = os.path.join(gcam_link_out_folder, file_name)
    #         cmd_str = f'cp -f {cam_path} {link_path}'
    #         # cmd_str = f'ln -s {cam_path} {link_path}'
    #         logger.info(cmd_str)
    #         os.system(cmd_str)

    # for idx_fold in range(num_fold):
    #     pred_result_csv = os.path.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
    #     cam_folder = os.path.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.{layer_flag}')
    #     pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
    #     file_list = pred_result_df['file_name']
    #
    #     for file_name in file_list:
    #         atlas_intensity_path = os.path.join(atlas_intensity_folder, file_name)
    #         atlas_jacobian_path = os.path.join(atlas_jacobian_folder, file_name)
    #         cam_path = os.path.join(cam_folder, file_name)

            # plot_obj = ClipPlotSeriesWithBack(
            #     cam_path,
            #     None,
            #     atlas_intensity_path,
            #     10, 35, 15,
            #     5,
            #     0, 1,
            #     -1000, 600,
            #     None
            # )

            # plot_obj = ClipPlotIntensityDeformationWall(
            #     atlas_intensity_path,
            #     atlas_jacobian_path,
            #     cam_path,
            #     10, 35, 15,
            #     5,
            #     -1000, 600,
            #     -0.75, 0.75,
            #     0, 1
            # )
            #
            # plot_obj.clip_plot(gcam_overlay_out_folder)


if __name__ == '__main__':
    main()