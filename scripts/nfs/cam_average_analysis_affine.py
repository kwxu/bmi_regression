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


affine_img_dir = '/nfs/masi/xuk9/SPORE/CAC_class/data/affine/s2_no_nan'
file_list_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/complete_list'

out_folder = '/nfs/masi/xuk9/SPORE/CAC_class/data/affine/s2_no_nan_average'
mkdir_p(out_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_0_cam.yaml')
    args = parser.parse_args()

    file_list = read_file_contents_list(file_list_txt)
    # folder_obj = DataFolder(ori_img_dir, file_list)
    folder_obj = DataFolder(affine_img_dir, file_list)

    ave_obj = AverageValidRegion(folder_obj, 10)
    ave_obj.run_get_average()
    ave_obj.output_result_folder(out_folder, ambient_val=-1000)


if __name__ == '__main__':
    main()