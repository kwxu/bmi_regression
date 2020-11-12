import os.path as osp
import os
import numpy as np
from src.tools.utils import get_logger
import argparse
import yaml
import pandas as pd
from src.tools.clinical import ClinicalDataReaderSPORE
import nibabel as nib
from src.tools.data_io import ScanWrapper
from sklearn.metrics import roc_curve
from src.tools.utils import mkdir_p


logger = get_logger('grad_cam_analysis')


def get_top_n_disagreement_df(pred_result_df, clinical_reader, n_top):
    # 1. Add the ground truth BMI
    file_list = pred_result_df['subject_name'].to_list()
    bmi_array, _ = clinical_reader.get_gt_value_BMI(file_list)
    pred_result_df['BMI'] = bmi_array

    sort_bmi_df = pred_result_df.sort_values(by=['BMI'])
    sort_pred_df = pred_result_df.sort_values(by=['pred'])

    sort_bmi_file_list = sort_bmi_df['subject_name'].to_list()
    sort_pred_file_list = sort_pred_df['subject_name'].to_list()

    rank_gap_list = np.zeros((len(file_list),), dtype=int)
    for idx_file in range(len(file_list)):
        file_name = file_list[idx_file]
        idx_bmi_sorted = sort_bmi_file_list.index(file_name)
        idx_pred_sorted = sort_pred_file_list.index(file_name)
        rank_gap_list[idx_file] = np.abs(idx_bmi_sorted - idx_pred_sorted)

    pred_result_df['RankGap'] = rank_gap_list
    ranked_gap_df = pred_result_df.sort_values(by=['RankGap'], ascending=False)
    ranked_gap_df = ranked_gap_df[["subject_name", "BMI", "pred", "RankGap"]]
    ranked_gap_df = ranked_gap_df.head(n_top)

    print(ranked_gap_df)

    return ranked_gap_df


def get_cam_file_path_list_0_1(cam_folder, file_name_list):
    cam_file_path_list_0 = [
        osp.join(cam_folder, file_name.replace('.nii.gz', f'_0.nii.gz'))
        for file_name in file_name_list
    ]
    cam_file_path_list_1 = [
        osp.join(cam_folder, file_name.replace('.nii.gz', f'_1.nii.gz'))
        for file_name in file_name_list
    ]

    return cam_file_path_list_0, cam_file_path_list_1


def get_average_map(file_list, save_path):
    first_img = ScanWrapper(file_list[0])
    im_shape = first_img.get_shape()
    sum_map = np.zeros(im_shape, dtype=float)

    for idx_image in range(len(file_list)):
        img = ScanWrapper(file_list[idx_image])
        img_data = img.get_data()
        print(f'Adding {file_list[idx_image]} ({idx_image} / {len(file_list)})')
        print(f'Max intensity {np.max(img_data)}')
        sum_map += img_data

    average_map = sum_map / float(len(file_list))
    print(f'Average map max int {np.max(average_map)}')
    first_img.save_scan_same_space(save_path, average_map)


def save_average(cam_list_dict, out_folder):
    mkdir_p(out_folder)

    for flag in cam_list_dict:
        # print(cam_list_dict)
        file_path_list = cam_list_dict[flag]
        print(f'{flag}: {len(file_path_list)}')
        out_file = osp.join(out_folder, f'{flag}.nii.gz')
        # get_average_map(file_path_list, out_file)


def get_split_file_list(split_ratio_std, result_df):
    pred_list = result_df['pred']
    target_list = result_df['target']

    diff_list = pred_list - target_list

    result_df['diff'] = diff_list

    diff_std = np.std(np.array(diff_list))
    diff_mean = np.mean(np.array(diff_list))

    outlier_df = result_df[
        (result_df['diff'] > (diff_mean + split_ratio_std * diff_std)) |
        (result_df['diff'] < (diff_mean - split_ratio_std * diff_std))
    ]

    normal_df = result_df[
        (result_df['diff'] <= (diff_mean + split_ratio_std * diff_std)) &
        (result_df['diff'] >= (diff_mean - split_ratio_std * diff_std))
    ]

    result_dict = {
        'outlier_list': outlier_df['file_path'].to_list(),
        'normal_list': normal_df['file_path'].to_list()
    }

    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_3_cam.yaml')
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    num_fold = config['fold_num']
    exp_dir = config['exp_dir']

    pred_result_df_list = []
    for idx_fold in range(num_fold):
        pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
        cam_folder = osp.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.layer2')
        file_list = pred_result_df['file_name'].to_list()
        file_path_list = [osp.join(cam_folder, file_name) for file_name in file_list]
        pred_result_df['file_path'] = file_path_list

        pred_result_df_list.append(pred_result_df)

    pred_result_total_df = pd.concat(pred_result_df_list, ignore_index=True)
    print(len(pred_result_total_df))

    cam_analysis_folder = osp.join(exp_dir, f'cam_analysis')
    mkdir_p(cam_analysis_folder)

    # result_dict = get_split_file_list(0.84, pred_result_total_df)
    # out_root = osp.join(cam_analysis_folder, 'split_60')

    result_dict = get_split_file_list(1.44, pred_result_total_df)
    out_root = osp.join(cam_analysis_folder, 'split_85')

    mkdir_p(out_root)
    out_average_outlier_path = osp.join(out_root, 'average_outlier.nii.gz')
    out_average_normal_path = osp.join(out_root, 'average_normal.nii.gz')

    get_average_map(result_dict['outlier_list'], out_average_outlier_path)
    get_average_map(result_dict['normal_list'], out_average_normal_path)


if __name__ == '__main__':
    main()

