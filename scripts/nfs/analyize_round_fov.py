import os
import os.path as osp
from src.tools.utils import get_logger
from tools.utils import mkdir_p
from src.tools.plot import ClipPlotSeriesWithBack
from src.tools.utils import read_file_contents_list, save_file_contents_list
from src.tools.data_io import ScanWrapper
import numpy as np


in_native_folder = '/nfs/masi/xuk9/SPORE/CAC_class/data/s14_ori_final_resample'
file_list_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/result_temporal'

out_folder_axial_clip = '/nfs/masi/xuk9/SPORE/CAC_class/axial_clip_native'
mkdir_p(out_folder_axial_clip)


def axial_clip_plot_native():
    file_name_list = read_file_contents_list(file_list_txt)
    for file_name in file_name_list:
        in_img_path = os.path.join(in_native_folder, file_name)
        cliper_obj = ClipPlotSeriesWithBack(
            in_img_path,
            None,
            None,
            10, 35, 15,
            1,
            -3000, 1000,
            None, None,
            None
        )
        cliper_obj.clip_plot_img_only(out_folder_axial_clip)


fov_file_list_folder = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/fov'
mkdir_p(fov_file_list_folder)
out_file_list_round_fov = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/fov/round_fov'
out_file_list_normal_fov = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/fov/normal_fov'

threshold_val = -2500
thres_num_voxel = 100

def get_list_round_and_normal_fov():
    file_name_list = read_file_contents_list(file_list_txt)
    round_list = []
    normal_list = []
    for file_name in file_name_list:
        img_obj = ScanWrapper(os.path.join(in_native_folder, file_name))
        img_data = img_obj.get_data()
        img_thres_data = (img_data < threshold_val).astype(int)
        if np.sum(img_thres_data) > thres_num_voxel:
            print(f'Round FOV: {file_name}')
            round_list.append(file_name)
        else:
            print(f'Normal FOV: {file_name}')
            normal_list.append(file_name)

    print(f'# Total round FOV: {len(round_list)}')
    print(f'# Total normal FOV: {len(normal_list)}')
    save_file_contents_list(out_file_list_round_fov, round_list)
    save_file_contents_list(out_file_list_normal_fov, normal_list)


def main():
    # axial_clip_plot_native()
    get_list_round_and_normal_fov()


if __name__ == '__main__':
    main()