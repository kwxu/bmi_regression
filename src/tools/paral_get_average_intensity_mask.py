import argparse
from tools.data_io import DataFolder, ScanWrapper
from tools.utils import get_logger
from tools.paral import AbstractParallelRoutine
import numpy as np


logger = get_logger('Mean intensity in mask')


class MeanIntensityMask(AbstractParallelRoutine):
    def __init__(self,
                 in_ori_folder_obj,
                 in_mask_folder_obj,
                 lung_label_list,
                 num_process):
        super().__init__(in_ori_folder_obj, num_process)
        self._in_mask_folder_obj = in_mask_folder_obj
        self._lung_label_list = lung_label_list

    def _run_single_scan(self, idx):
        in_img = ScanWrapper(self._in_data_folder.get_file_path(idx))
        in_mask = ScanWrapper(self._in_mask_folder_obj.get_file_path(idx))

        in_img_data = in_img.get_data()
        in_mask_data = in_mask.get_data()

        mask_data = np.zeros(in_mask_data.shape, dtype=int)
        for lung_label in self._lung_label_list:
            mask_match_map = (in_mask_data == lung_label).astype(int)
            print(f'lung label: {lung_label}')
            print(f'match_map shape: {mask_match_map.shape}')
            print(f'num posi voxels: {np.sum(mask_match_map)}')
            mask_data += mask_match_map

        mask_data = ~mask_data.astype('bool')
        print(f'Final mask: {np.sum(mask_data)}')
        in_img_data_with_mask = np.ma.array(in_img_data, mask=mask_data.astype('bool'))

        mean_in_mask = in_img_data_with_mask.mean()
        file_name = self._in_data_folder.get_file_name(idx)

        result_dict = {
            'file_name': file_name,
            'mean': mean_in_mask
        }

        print(result_dict)

        return result_dict
