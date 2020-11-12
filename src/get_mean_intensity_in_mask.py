from tools.data_io import DataFolder, ScanWrapper
from tools.utils import read_file_contents_list
from tools.paral_get_average_intensity_mask import MeanIntensityMask
from tools.clinical import ClinicalDataReaderSPORE
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress
import os
import matplotlib.pyplot as plt



in_clinical_csv = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/label_full_combined.csv'


def get_csv(args):
    file_list = read_file_contents_list(args.file_list_txt)
    in_ori_folder_obj = DataFolder(args.in_ori_folder, file_list)
    in_mask_folder_obj = DataFolder(args.in_mask_folder, file_list)

    exe_obj = MeanIntensityMask(in_ori_folder_obj, in_mask_folder_obj, [2, 4], 20)
    result_dict_list = exe_obj.run_parallel()

    result_df = pd.DataFrame(result_dict_list)
    result_df = result_df.set_index('file_name')

    print(f'Output csv to {args.out_csv}')
    result_df.to_csv(args.out_csv)


def analysis_correlation(args):
    result_df = pd.read_csv(args.out_csv)
    result_df = result_df.set_index('file_name')

    file_list = read_file_contents_list(args.file_list_txt)
    clinical_reader = ClinicalDataReaderSPORE.create_spore_data_reader_csv(in_clinical_csv)
    bmi_array, valid_file_name_list = clinical_reader.get_gt_value_BMI(file_list)

    valid_result_df = result_df.loc[valid_file_name_list]
    # valid_result_df['bmi'] = bmi_array

    valid_mean_list = valid_result_df['mean'].to_numpy()

    print(pearsonr(bmi_array, valid_mean_list))

    slope, intercept, r_value, p_value, std_err = linregress(bmi_array, valid_mean_list)
    reg_val = intercept + slope * bmi_array

    out_png = os.path.join('/nfs/masi/xuk9/SPORE/CAC_class/data', 'bmi_mean_lung.png')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(bmi_array, valid_mean_list, label=f'Samples')
    ax.plot(bmi_array, reg_val, color='r', label=f'Slope={slope:.3f}, p-value={p_value:.3E}')
    ax.set_xlabel('BMI ($kg/m^2$)')
    ax.set_ylabel('Averaged intensity (HU) in lung region')

    ax.legend(loc='best')

    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-ori-folder', type=str, default='/nfs/masi/xuk9/SPORE/CAC_class/data/s1_resampled')
    parser.add_argument('--in-mask-folder', type=str, default='/nfs/masi/xuk9/SPORE/CAC_class/data/s2_lung_mask')
    parser.add_argument('--out-csv', type=str, default='/nfs/masi/xuk9/SPORE/CAC_class/data/bmi_mean_lung.csv')
    parser.add_argument('--file-list-txt', type=str, default='/nfs/masi/xuk9/SPORE/CAC_class/file_lists/complete_list')
    args = parser.parse_args()

    # get_csv(args)
    analysis_correlation(args)

    # file_list = read_file_contents_list(args.file_list_txt)
    # in_ori_folder_obj = DataFolder(args.in_ori_folder, file_list)
    # in_mask_folder_obj = DataFolder(args.in_mask_folder, file_list)
    #
    # exe_obj = MeanIntensityMask(in_ori_folder_obj, in_mask_folder_obj, [2, 4], 10)
    # result_dict_list = exe_obj.run_parallel()
    #
    # result_df = pd.DataFrame(result_dict_list)
    # result_df = result_df.set_index('file_name')
    #
    # clinical_reader = ClinicalDataReaderSPORE.create_spore_data_reader_csv(in_clinical_csv)
    # bmi_array, valid_file_name_list = clinical_reader.get_gt_value_BMI(file_list)
    #
    # valid_result_df = result_df.loc[valid_file_name_list]
    # lung_mean_array = valid_result_df['mean'].to_numpy()
    #
    # corr_val = np.corrcoef(lung_mean_array, bmi_array)




if __name__ == '__main__':
    main()