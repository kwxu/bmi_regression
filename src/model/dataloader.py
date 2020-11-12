from model.imgloader_CT_clss_3D import pytorch_loader_clss3D
from torch.utils.data import DataLoader
from model.dataset import get_train_data_dict
import os
from tools.cross_validation import get_idx_list_array_n_fold_regression_bl
import numpy as np


def get_data_loader_cv(config):
    fold_num = config['fold_num']
    batch_size = config['batch_size']
    # num_workers = 2
    num_workers = 4

    train_valid_data_dict = get_train_data_dict(config)
    train_valid_file_name_list = [os.path.basename(file_path) for file_path in train_valid_data_dict['img_files']]

    fold_train_idx_list_array, fold_validate_idx_list_array, fold_test_idx_list_array = \
        get_idx_list_array_n_fold_regression_bl(train_valid_file_name_list, fold_num)

    def create_data_loader_list(fold_idx_list_array, shuffle):
        data_loader_list = []
        for idx_fold in range(fold_num):
            fold_idx_list = fold_idx_list_array[idx_fold]
            data_dict = {}
            for key_str in train_valid_data_dict:
                data_dict[key_str] = [train_valid_data_dict[key_str][idx] for idx in fold_idx_list]

            data_set = pytorch_loader_clss3D(
                data_dict,
                config
            )

            print(f'Demographic of fold {idx_fold}:')
            bins = np.array([0, 18.5, 25, 30, 100])
            inds, bin_edges = np.histogram(np.array(data_dict['gt_val']), bins)
            for bin_idx in range(len(bins) - 1):
                print(f'[{bins[bin_idx]}, {bins[bin_idx + 1]}]: {inds[bin_idx]}')

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )
            data_loader_list.append(data_loader)

        return data_loader_list

    train_data_loader_list = create_data_loader_list(fold_train_idx_list_array, True)
    validate_data_loader_list = create_data_loader_list(fold_validate_idx_list_array, False)
    test_data_loader_list = create_data_loader_list(fold_test_idx_list_array, False)

    return train_data_loader_list, validate_data_loader_list, test_data_loader_list



