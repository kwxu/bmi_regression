from src.tools.cross_validation import get_idx_list_array_bmi_session_level_split
from src.tools.utils import read_file_contents_list, mkdir_p
import pandas as pd
from src.tools.clinical import ClinicalDataReaderSPORE
import os
import matplotlib.pyplot as plt


in_label_file = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/label_full_combined.csv'
in_file_list_file = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/result_temporal'
num_fold = 5
out_folder = '/nfs/masi/xuk9/SPORE/CAC_class/test_output/k_fold_split'
mkdir_p(out_folder)


def main():
    file_list = read_file_contents_list(in_file_list_file)
    label_reader_obj = ClinicalDataReaderSPORE.create_spore_data_reader_csv(in_label_file)
    bmi_list, valid_file_list = label_reader_obj.get_gt_value_BMI(file_list)
    # print(bmi_list[:10])
    train_idx_fold_array, valid_idx_fold_array, test_idx_fold_array = get_idx_list_array_bmi_session_level_split(
        bmi_list,
        num_fold
    )

    test_bmi_list_array = []
    for idx_fold in range(num_fold):
        test_bmi_list = [bmi_list[data_idx] for data_idx in test_idx_fold_array[idx_fold]]
        test_bmi_list_array.append(test_bmi_list)

    out_png = os.path.join(out_folder, 'fold_distribution.png')
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.hist(
        test_bmi_list_array,
        bins=5,
        label=[f'fold-{idx}' for idx in range(5)],
        alpha=0.7,
        rwidth=0.8,
        density=True
    )
    ax.legend(loc='best')
    plt.grid(axis='y', alpha=0.8)
    plt.xlabel('BMI')
    plt.ylabel('Density (Count / Total)')

    print(f'Save image to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == '__main__':
    main()

