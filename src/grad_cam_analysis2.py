import os.path as osp
import os
import numpy as np
from tools.utils import get_logger
import argparse
import yaml
import pandas as pd
from tools.clinical import ClinicalDataReaderSPORE
import nibabel as nib
from tools.data_io import ScanWrapper
from sklearn.metrics import roc_curve
from tools.utils import mkdir_p


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


def get_optimal_thres(pred_result_df, in_folder):
    # 1. Get the threshold by maximizing the G-means
    pred_list = pred_result_df['pred'].to_numpy()
    gt_list = pred_result_df['label'].to_numpy().astype(int)

    fpr, tpr, thres = roc_curve(gt_list, pred_list)
    g_means = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(g_means)
    optimal_thres = thres[ix]

    TP_df = pred_result_df[pred_result_df['label'] & (pred_result_df['pred'] > optimal_thres)]
    FN_df = pred_result_df[pred_result_df['label'] & (pred_result_df['pred'] <= optimal_thres)]
    TN_df = pred_result_df[(~pred_result_df['label']) & (pred_result_df['pred'] <= optimal_thres)]
    FP_df = pred_result_df[(~pred_result_df['label']) & (pred_result_df['pred'] > optimal_thres)]

    print(f'Thres: {optimal_thres:.3f}')
    print(f'TP: {len(TP_df)}')
    print(f'FN: {len(FN_df)}')
    print(f'TN: {len(TN_df)}')
    print(f'FP: {len(FP_df)}')

    result_dict = {
        'optimal_thres': optimal_thres,
        'TP_file_list': TP_df['subject_name'].to_list(),
        'FN_file_list': FN_df['subject_name'].to_list(),
        'TN_file_list': TN_df['subject_name'].to_list(),
        'FP_file_list': FP_df['subject_name'].to_list()
    }

    return result_dict


def append_cam_files(cam_list_dict, cam_folder, slot_idx, thres_result):
    new_TP_list = [
        osp.join(cam_folder, file_name.replace('.nii.gz', f'_{slot_idx}.nii.gz'))
        for file_name in thres_result['TP_file_list']
    ]
    new_FN_list = [
        osp.join(cam_folder, file_name.replace('.nii.gz', f'_{slot_idx}.nii.gz'))
        for file_name in thres_result['FN_file_list']
    ]
    new_TN_list = [
        osp.join(cam_folder, file_name.replace('.nii.gz', f'_{slot_idx}.nii.gz'))
        for file_name in thres_result['TN_file_list']
    ]
    new_FP_list = [
        osp.join(cam_folder, file_name.replace('.nii.gz', f'_{slot_idx}.nii.gz'))
        for file_name in thres_result['FP_file_list']
    ]

    cam_list_dict['TP_list'] += new_TP_list
    cam_list_dict['FN_list'] += new_FN_list
    cam_list_dict['TN_list'] += new_TN_list
    cam_list_dict['FP_list'] += new_FP_list

    return cam_list_dict


def save_average(cam_list_dict, out_folder):
    mkdir_p(out_folder)

    for flag in cam_list_dict:
        # print(cam_list_dict)
        file_path_list = cam_list_dict[flag]
        print(f'{flag}: {len(file_path_list)}')
        out_file = osp.join(out_folder, f'{flag}.nii.gz')
        # get_average_map(file_path_list, out_file)


cam_folder_nfs = '/nfs/masi/xuk9/SPORE/CAC_class/average_cam'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_3.6_nfs.yaml')
    parser.add_argument('--cam-folder', type=str, default=cam_folder_nfs)
    args = parser.parse_args()

    mkdir_p(args.cam_folder)

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    num_fold = config['fold_num']
    exp_dir = config['exp_dir']
    layer_flag = config['gcam_target_layer']

    file_path_list_array = []
    for idx_fold in range(num_fold):
    # for idx_fold in range(0, 1):
        pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
        cam_folder = osp.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.{layer_flag}')
        mkdir_p(cam_folder)
        pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
        file_list = pred_result_df['file_name']
        file_path_list = [osp.join(cam_folder, file_name)
                          for file_name in file_list]

        # out_average_path = osp.join(exp_dir, f'fold_{idx_fold}/cam_average_layer2.nii.gz')
        # get_average_map(file_path_list, out_average_path)
        file_path_list_array += file_path_list

    # cam_analysis_folder = osp.join(exp_dir, f'cam_analysis')
    # mkdir_p(cam_analysis_folder)
    # out_average_path = osp.join(cam_analysis_folder, 'averaged_all.nii.gz')
    out_average_path = osp.join(args.cam_folder, f'{args.yaml_config}.{layer_flag}.nii.gz')
    get_average_map(file_path_list_array, out_average_path)

    #
    # cam_list_dict_1 = {
    #     'TP_list': [],
    #     'FN_list': [],
    #     'TN_list': [],
    #     'FP_list': []
    # }
    #
    # cam_list_dict_0 = {
    #     'TP_list': [],
    #     'FN_list': [],
    #     'TN_list': [],
    #     'FP_list': []
    # }
    #
    # for idx_fold in range(num_fold):
    # # for idx_fold in range(0, 1):
    #     pred_result_csv = osp.join(exp_dir, f'fold_{idx_fold}/test/predict.csv')
    #     cam_folder = osp.join(exp_dir, f'fold_{idx_fold}/grad_CAM/test.layer3')
    #     pred_result_df = pd.read_csv(pred_result_csv, index_col=False)
    #     result_dict = get_optimal_thres(pred_result_df, cam_folder)
    #
    #     cam_list_dict_0 = append_cam_files(cam_list_dict_0, cam_folder, 0, result_dict)
    #     cam_list_dict_1 = append_cam_files(cam_list_dict_1, cam_folder, 1, result_dict)
    #
    # out_folder_0 = osp.join(exp_dir, f'cam_analysis/cam_average_0')
    # out_folder_1 = osp.join(exp_dir, f'cam_analysis/cam_average_1')
    # # save_average(cam_list_dict_0, out_folder_0)
    # save_average(cam_list_dict_1, out_folder_1)


if __name__ == '__main__':
    main()

