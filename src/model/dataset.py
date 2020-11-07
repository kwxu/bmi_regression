from tools.clinical import ClinicalDataReaderSPORE
from tools.utils import read_file_contents_list
from tools.data_io import DataFolder
from tools.utils import get_logger
import numpy as np


logger = get_logger('Dataset')


def get_data_dict(config, file_list_txt):
    task = config['task']
    in_folder = config['input_img_dir']
    label_csv = config['label_csv']

    in_folder_obj = DataFolder(in_folder, read_file_contents_list(file_list_txt))
    file_list = in_folder_obj.get_data_file_list()

    clinical_reader = ClinicalDataReaderSPORE.create_spore_data_reader_csv(label_csv)

    label_array = None
    file_list_with_valid_label = None

    if task == 'BMI':
        label_array, file_list_with_valid_label = clinical_reader.get_gt_value_BMI(file_list)

    subject_list = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
                    for file_name in file_list_with_valid_label]

    in_folder_obj.set_file_list(file_list_with_valid_label)
    file_path_list = in_folder_obj.get_file_path_list()

    data_dict = {
        'img_names': file_list_with_valid_label,
        'img_subs': subject_list,
        'img_files': file_path_list,
        'gt_val': label_array
    }

    if config['add_jacobian_map']:
        in_jacobian_folder = config['input_jac_dir']
        in_jacobian_folder_obj = DataFolder(in_jacobian_folder, file_list_with_valid_label)
        jacobian_map_path_list = in_jacobian_folder_obj.get_file_path_list()
        data_dict['jacobian_maps'] = jacobian_map_path_list

    return data_dict


def get_train_data_dict(config):
    train_file_list_txt = config['train_file_list_txt']

    train_data_dict = get_data_dict(config, train_file_list_txt)
    train_val_list = train_data_dict['gt_val'].tolist()

    # BMI range bins
    bins = np.array([0, 18.5, 25, 30, 100])
    inds, bin_edges = np.histogram(np.array(train_val_list), bins)

    logger.info(f'Character Summary of Dataset:')
    for bin_idx in range(len(bins) - 1):
        logger.info(f'[{bins[bin_idx]}, {bins[bin_idx + 1]}]: {inds[bin_idx]}')

    return train_data_dict

