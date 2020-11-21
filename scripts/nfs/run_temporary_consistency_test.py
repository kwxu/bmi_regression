import pandas as pd
import numpy as np
import argparse
import os
import os.path as osp
from src.tools.utils import get_logger
import yaml
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tools.utils import mkdir_p
from src.tools.clinical import ClinicalDataReaderSPORE
from tools.utils import read_file_contents_list, save_file_contents_list


logger = get_logger('Run temporary consistency test')


in_file_list_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/complete_list'
in_csv_file = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/label_full_combined.csv'
in_raw_label_file_xlsx = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/clinical/label.xlsx'

out_height_weight_added_csv = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/label_add_H_W.csv'
valid_bmi_file_list = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/bmi_valid'

out_diff_hist_png = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/diff_hist.png'

out_long_exlude_list = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/bmi_exclude_long_sess_list.txt'
out_include_bmi_list = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/bmi_include_list.txt'
out_exclude_bmi_list = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/bmi_exclude_list.txt'


def hist_plot_with_95_percentile(score_list, percentile_pos, percentile_val, out_png):
    f, ax = plt.subplots(figsize=(8, 6))
    ax.hist(score_list, bins=40, rwidth=0.8)

    print(f'Plot consistency score with {percentile_pos:d}th percentile')
    print(f'{percentile_val:.3f}')

    ax.axvline(x=percentile_val, c='r', alpha=0.6, linestyle='--',
               label=f'{percentile_pos:d}th percentile ({percentile_val:.3f})')
    ax.legend(loc='best')

    ax.set_xlabel(f'Inconsistency score (minimal absolute difference to the rest sessions of the same subject)')
    ax.set_ylabel(f'Count')
    # ax.set_xlim(0, 1)

    logger.info(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def add_weight_height_to_df():
    label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_csv(in_csv_file)
    raw_label_df = pd.read_excel(in_raw_label_file_xlsx)

    label_obj.get_attributes_from_original_label_file(raw_label_df, 'heightinches')
    label_obj.get_attributes_from_original_label_file(raw_label_df, 'weightpounds')

    label_obj._df.to_csv(out_height_weight_added_csv)


def analyze_the_temporal_consistency_check(attr_flag):
    # Analayiss
    label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_csv(out_height_weight_added_csv)

    file_name_list = read_file_contents_list(valid_bmi_file_list)
    inconsistency_data_dict = label_obj.temporal_consistency_check(attr_flag, file_name_list)

    out_png = os.path.join('/nfs/masi/xuk9/SPORE/CAC_class/clinical', f'inconsistency_hist_{attr_flag}.png')

    inconsistency_list = np.array([inconsistency_data_dict[sess]['inconsistent_score']
                                   for sess in inconsistency_data_dict])

    percentile_pos = 95

    percentile_val = np.percentile(inconsistency_list, percentile_pos)

    hist_plot_with_95_percentile(inconsistency_list, percentile_pos, percentile_val, out_png)

    # Return the inconsistent session name list

    sess_list = [sess for sess in inconsistency_data_dict]
    inconsistency_idx_list = np.argwhere(inconsistency_list > percentile_val)[:, 0]
    inconsist_sess = [sess_list[idx] for idx in inconsistency_idx_list]

    return inconsist_sess, percentile_val


def filtering(total_sess_list):
    save_file_contents_list(
        out_long_exlude_list,
        [sess_name + '.nii.gz' for sess_name in total_sess_list]
    )

    print(f'# Total inconsistency sess: {len(total_sess_list)}')

    print(f'How many cases left:')
    file_name_list = read_file_contents_list(valid_bmi_file_list)
    file_name_no_ext = [
        file_name.replace('.nii.gz', '')
        for file_name in file_name_list
    ]
    all_subj_list = ClinicalDataReaderSPORE._get_subj_list_from_sess_list(file_name_no_ext)

    long_sess_list = ClinicalDataReaderSPORE._get_longitudinal_sess_list(file_name_no_ext)

    left_long_sess_list = [sess_name for sess_name in long_sess_list if sess_name not in total_sess_list]
    left_subj_list = ClinicalDataReaderSPORE._get_subj_list_from_sess_list(left_long_sess_list)

    print(f'# left sessions: {len(left_long_sess_list)} ({len(file_name_list)})')
    print(f'# left subjects: {len(left_subj_list)} ({len(all_subj_list)})')

    file_name_include_total = [sess_name + '.nii.gz' for sess_name in left_long_sess_list]

    save_file_contents_list(
        out_include_bmi_list,
        file_name_include_total
    )

    file_excluded_total = [sess_name + '.nii.gz' for sess_name in file_name_no_ext if sess_name not in left_long_sess_list]
    save_file_contents_list(
        out_exclude_bmi_list,
        file_excluded_total
    )

    return file_name_include_total, file_excluded_total


out_distribution_compare_png = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/all_include_distribution_compare.png'


def get_bmi_distribution_compare(file_name_list_include, file_name_list_total):
    label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_csv(out_height_weight_added_csv)
    include_bmi_list = label_obj.get_value_list('bmi', file_name_list_include)
    total_bmi_list = label_obj.get_value_list('bmi', file_name_list_total)

    data_array_sequence = []
    data_array_sequence.append(include_bmi_list)
    data_array_sequence.append(total_bmi_list)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        data_array_sequence,
        # bins='auto',
        bins=10,
        color=['r', 'b'],
        label=[f'Filtered ({len(file_name_list_include)})', f'All ({len(file_name_list_total)})'],
        alpha=0.8,
        rwidth=0.9,
        density=True
    )
    ax.legend(loc='best')
    plt.grid(axis='y', alpha=0.8)
    plt.xlabel('BMI')
    plt.ylabel('Density (Count / Total)')

    logger.info(f'Save image to {out_distribution_compare_png}')
    plt.savefig(out_distribution_compare_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def get_num_longitudinal_sessions_in_raw_label_data():
    file_name_list = read_file_contents_list(valid_bmi_file_list)
    subject_id_list = [
        ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
        for file_name in file_name_list
    ]
    subject_id_spore_format_list = [
        f'SPORE_{subj_id:08d}'
        for subj_id in subject_id_list
    ]

    subject_id_spore_format_list = list(set(subject_id_spore_format_list))

    # 1. Get the number of subject that have longitudinal in raw label file.
    # 2. Get the number of sessions that have ...

    raw_label_df = pd.read_excel(in_raw_label_file_xlsx)
    long_subj_list = []
    # print(raw_label_df['SPORE'].to_list()[:10])
    for subj_id_spore_format in subject_id_spore_format_list:
        subj_df = raw_label_df[raw_label_df['SPORE'] == subj_id_spore_format]
        if len(subj_df) > 1:
            long_subj_list.append(subj_id_spore_format)

    print(long_subj_list[:10])
    print(len(long_subj_list))


def main():
    # add_weight_height_to_df()

    # inconsist_list_H, thres_H = analyze_the_temporal_consistency_check('heightinches')
    # inconsist_list_W, thres_W = analyze_the_temporal_consistency_check('weightpounds')
    # inconsist_list_bmi, thres_bmi = analyze_the_temporal_consistency_check('bmi')
    #
    # print(f'# Height inconsistency: {len(inconsist_list_H)}, thres: {thres_H:.3f}')
    # print(f'# Weight inconsistency: {len(inconsist_list_W)}, thres: {thres_W:.3f}')
    # print(f'# BMI inconsistency: {len(inconsist_list_bmi)}, thres: {thres_bmi:.3f}')
    #
    # # total_sess_list = list(set(inconsist_list_H + inconsist_list_W + inconsist_list_bmi))
    # # total_sess_list = list(set(inconsist_list_H + inconsist_list_W))
    # total_sess_list = inconsist_list_bmi
    # file_name_include, file_name_excluded = filtering(total_sess_list)
    # get_bmi_distribution_compare(file_name_include, file_name_include + file_name_excluded)

    get_num_longitudinal_sessions_in_raw_label_data()


if __name__ == '__main__':
    main()