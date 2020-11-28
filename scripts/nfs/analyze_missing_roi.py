import os
import os.path as osp
from src.tools.utils import get_logger
from tools.utils import mkdir_p
from src.tools.plot import ClipPlotSeriesWithBack
from src.tools.utils import read_file_contents_list, save_file_contents_list, get_logger
from src.tools.data_io import ScanWrapper
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tools.plot import mean_diff_plot, scatter_plot


logger = get_logger('Analyze missing ROI')


in_body_mask = '/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/atlas_body_mask/body_seg_resampled.nii.gz'
body_mask_overlap_roi_folder = '/nfs/masi/xuk9/SPORE/CAC_class/data/atlas/valid_region/s6_body_mask_intersect'
file_list_txt = '/nfs/masi/xuk9/SPORE/CAC_class/file_lists/result_temporal'

out_analyze_missing_roi_folder = '/nfs/masi/xuk9/SPORE/CAC_class/clinical/missing_roi'
mkdir_p(out_analyze_missing_roi_folder)

out_missing_ratio_csv = os.path.join(out_analyze_missing_roi_folder, 'missing_ratio.csv')


def _get_sess_missing_ratio(file_name):

    body_mask_data = ScanWrapper(in_body_mask).get_data()
    in_mask_data = ScanWrapper(os.path.join(body_mask_overlap_roi_folder, file_name)).get_data()

    missing_ratio = 1 - (np.sum(in_mask_data.astype(int))) / (np.sum(body_mask_data.astype(int)))

    return missing_ratio


def get_missing_roi_ratio():
    file_name_list = read_file_contents_list(file_list_txt)

    data_dict = []
    num_total = len(file_name_list)
    counter = 0
    for file_name in file_name_list:
        logger.info(f'Analyze {file_name} ({counter} / {num_total})')
        missing_ratio = _get_sess_missing_ratio(file_name)
        logger.info(f'missing ratio - {missing_ratio:.3f}')
        counter += 1

        data_dict.append(
            {
                'file_name': file_name,
                'missing_ratio': missing_ratio
            }
        )

    missing_ratio_df = pd.DataFrame(data_dict)
    missing_ratio_df.to_csv(out_missing_ratio_csv)


def plot_missing_roi_vs_rmse_hist(yaml_config):
    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    # num_fold = config['fold_num']
    exp_dir = config['exp_dir']

    missing_ratio_data_df = pd.read_csv(out_missing_ratio_csv)
    file_name_list = missing_ratio_data_df['file_name'].to_list()
    missing_ratio_list = missing_ratio_data_df['missing_ratio'].to_numpy()

    pred_df = pd.read_csv(os.path.join(exp_dir, 'pred_total.csv'), index_col='file_name')
    used_pred_df = pred_df.loc[file_name_list]
    used_pred_df['missing_ratio'] = missing_ratio_list

    # Split the dataset using missing ratio percentiles
    # perc_75 = np.percentile(missing_ratio_list, 75)
    perc_75 = np.percentile(missing_ratio_list, 85)
    perc_25 = np.percentile(missing_ratio_list, 25)

    perc_low_df = used_pred_df[used_pred_df['missing_ratio'] <= perc_25]
    perc_mid_df = used_pred_df.loc[(used_pred_df['missing_ratio'] > perc_25) &
                                   (used_pred_df['missing_ratio'] <= perc_75)]
    perc_up_df = used_pred_df[used_pred_df['missing_ratio'] > perc_75]

    shift_data_sequence = []
    shift_data_sequence.append(perc_low_df['diff'].to_numpy())
    shift_data_sequence.append(perc_mid_df['diff'].to_numpy())
    shift_data_sequence.append(perc_up_df['diff'].to_numpy())

    # shift_data_sequence.append(perc_low_df['diff_abs'].to_numpy())
    # shift_data_sequence.append(perc_mid_df['diff_abs'].to_numpy())
    # shift_data_sequence.append(perc_up_df['diff_abs'].to_numpy())

    bins = list(range(-10, 11, 2))

    f, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        shift_data_sequence,
        # bins=5,
        bins=bins,
        color=['r', 'b', 'y'],
        label=[
            f'missing ratio < {perc_25:.3f} ({len(perc_low_df)})',
            f'{perc_25:.3f} <= missing ratio < {perc_75:.3f} ({len(perc_mid_df)})',
            f'missing ratio >= {perc_75:.3f} ({len(perc_up_df)})'
        ],
        alpha=0.8,
        rwidth=0.9,
        density=True
    )

    ax.legend(loc='best')
    plt.grid(axis='y', alpha=0.8)
    plt.xlabel('Prediction - Ground-Truth Shift')
    plt.ylabel('Density (Count / Total)')

    out_png = os.path.join(exp_dir, 'missing_ratio_vs_shift1.png')
    print(f'Save image to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()



def main():
    yaml_config = 'simg_bmi_regression_3.6.4_nfs.yaml'
    # yaml_config = 'simg_bmi_regression_3.6.5_nfs.yaml'
    # yaml_config = 'simg_bmi_regression_0_3_nfs.yaml'
    # yaml_config = 'simg_bmi_regression_21.1_nfs.yaml'
    # plot_round_fov_output(yaml_config)
    # get_missing_roi_ratio()
    plot_missing_roi_vs_rmse_hist(yaml_config)


if __name__ == '__main__':
    main()
