from src.tools.paral_average import AverageValidRegion
from src.tools.data_io import ScanWrapper, DataFolder
import argparse
import yaml
import pandas as pd
import os.path as osp
import os
import numpy as np
from src.tools.utils import get_logger, read_file_contents_list, mkdir_p


logger = get_logger('grad_cam_analysis')


ori_img_dir = '/nfs/masi/xuk9/SPORE/CAC_class/data/s14_ori_final_resample'
file_list_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/complete_list'

ori_atlas_dir = '/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/ori/s6_resampled_full_atlas_roi_no_nan'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_0_cam.yaml')
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    num_fold = config['fold_num']
    exp_dir = config['exp_dir']

    out_dir = osp.join(exp_dir, 'cam_analysis/atlas_space')
    mkdir_p(out_dir)

    file_list = read_file_contents_list(file_list_txt)
    # folder_obj = DataFolder(ori_img_dir, file_list)
    folder_obj = DataFolder(ori_atlas_dir, file_list)

    ave_obj = AverageValidRegion(folder_obj, 10)
    ave_obj.run_get_average()
    ave_obj.output_result_folder(out_dir, ambient_val=-1000)


if __name__ == '__main__':
    main()