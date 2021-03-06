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

    if config['add_valid_mask_map'] | config['apply_random_valid_mask']:
        in_valid_mask_folder = config['input_valid_mask_dir']
        in_valid_mask_folder_obj = DataFolder(in_valid_mask_folder, file_list_with_valid_label)
        valid_mask_path_list = in_valid_mask_folder_obj.get_file_path_list()
        data_dict['valid_masks'] = valid_mask_path_list

    if config['add_d_index_map']:
        in_d_index_map_folder = config['input_d_index_dir']
        in_d_index_map_folder_obj = DataFolder(in_d_index_map_folder, file_list_with_valid_label)
        d_index_map_path_list = in_d_index_map_folder_obj.get_file_path_list()
        data_dict['d_index_maps'] = d_index_map_path_list

    if config['add_jac_elem_maps']:
        in_jac_elem_folder = config['input_jac_elem_dir']
        in_jac_elem_folder_obj = DataFolder(in_jac_elem_folder, file_list_with_valid_label)
        for idx_elem in range(9):
            in_jac_elem_path_list = [map_path.replace('.nii.gz', f'_{idx_elem}.nii.gz')
                                     for map_path in in_jac_elem_folder_obj.get_file_path_list()]
            data_dict[f'jac_elem_{idx_elem}_map'] = in_jac_elem_path_list

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

