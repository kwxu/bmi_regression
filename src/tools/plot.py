import matplotlib.pyplot as plt
import numpy as np
from tools.data_io import ScanWrapper
from matplotlib import colors
import os
from tools.utils import mkdir_p, get_logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
from sklearn.metrics import r2_score


logger = get_logger('Plot')


def mean_diff_plot(pred_list, gt_list, rmse_list, out_png):
    f, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.mean_diff_plot(pred_list, gt_list, ax=ax)
    ax.set_title(f'RMSE: {np.mean(rmse_list):.4f}, R2: {r2_score(gt_list, pred_list):.4f}')

    logger.info(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def scatter_plot(pred_list, gt_list, rmse_list, out_png, lower_bound=12, upper_bound=50):
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gt_list, pred_list, alpha=0.7)
    ax.set_xlabel(f'Ground-Truth BMI')
    ax.set_ylabel(f'Predicted BMI')
    ax.set_title(f'RMSE: {np.mean(rmse_list):.4f}, R2: {r2_score(gt_list, pred_list):.4f}')

    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle='--', alpha=0.7, c='r')

    logger.info(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_training_curve(data_dict, out_png):
    fig, ax = plt.subplots(figsize=(8, 6))

    train_loss = data_dict['train_loss']
    valid_loss = data_dict['valid_loss']
    start_epoch = data_dict['start_epoch']
    end_epoch = data_dict['end_epoch']
    loss_str = data_dict['loss_str']

    epoch_array = range(start_epoch, end_epoch)
    train_loss_plot = np.full((end_epoch,), fill_value=np.nan)
    valid_loss_plot = np.full((end_epoch,), fill_value=np.nan)

    end_idx_train = len(train_loss)
    end_idx_valid = len(valid_loss)
    if end_idx_train > end_epoch:
        end_idx_train = end_epoch
    if end_idx_valid > end_epoch:
        end_idx_valid = end_epoch

    train_loss_plot[start_epoch:end_idx_train] = train_loss[start_epoch:end_idx_train]
    valid_loss_plot[start_epoch:end_idx_valid] = valid_loss[start_epoch:end_idx_valid]

    ax.plot(epoch_array, train_loss_plot[start_epoch:], label='Training loss')
    ax.plot(epoch_array, valid_loss_plot[start_epoch:], label='Validation loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(loss_str)
    ax.set_xlim(start_epoch, end_epoch)
    ax.set_ylim(0, 30)
    ax.legend(loc='best')

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()



def plot_cv_roc(performance_array, png_path):
    fig, ax = plt.subplots(figsize=(18, 12))

    num_fold = len(performance_array)
    # ax.plot([0, 1], [0, 1], linestyle='--', color='b', lw=2, label='No skill', alpha=0.8)

    fpr_array = [valid_result['fpr'] for valid_result in performance_array]
    tpr_array = [valid_result['tpr'] for valid_result in performance_array]
    auc_array = [valid_result['roc_auc'] for valid_result in performance_array]

    # plot ROC of each fold
    for idx_fold in range(num_fold):
        ax.plot(fpr_array[idx_fold], tpr_array[idx_fold], lw=2,
                label=f'Fold {idx_fold + 1} (AUC = {auc_array[idx_fold]:.3f})', alpha=0.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic Curve on Testing Set')

    ax.legend(loc='best')

    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_class_pred(pred_df, out_png):
    # print(pred_df.label.to_list())
    gt_true_df = pred_df[pred_df.label]
    gt_false_df = pred_df[pred_df.label == False]

    gt_true_df = gt_true_df.sort_values(by=['pred'])
    gt_false_df = gt_false_df.sort_values(by=['pred'])

    gt_true_file_list = gt_true_df.subject_name.to_list()
    gt_false_file_list = gt_false_df.subject_name.to_list()

    gt_true_pred_val = gt_true_df.pred.to_list()
    gt_false_pred_val = gt_false_df.pred.to_list()

    show_num = 10

    print(f'Top {show_num} false negative:')
    for idx_file in range(show_num):
        file_name = gt_true_file_list[idx_file]
        pred_val = gt_true_pred_val[idx_file]
        print(f'{file_name}: {pred_val:.5f}')

    print(f'Top {show_num} false positive:')
    num_gt_false = len(gt_false_df)
    for idx_file in range(show_num):
        idx_use = num_gt_false - idx_file - 1
        file_name = gt_false_file_list[idx_use]
        pred_val = gt_false_pred_val[idx_use]
        print(f'{file_name}: {pred_val:.5f}')

    x_ticks = [0, 1]
    x_labels = ['False', 'True']

    fig, ax = plt.subplots(figsize=(18, 12))
    x_gt_false = np.zeros((len(gt_false_pred_val),))
    x_gt_false[:] = x_ticks[0]
    x_gt_true = np.zeros((len(gt_true_pred_val),))
    x_gt_true[:] = x_ticks[1]
    ax.scatter(x_gt_false, gt_false_pred_val, c='r')
    ax.scatter(x_gt_true, gt_true_pred_val, c='b')

    ax.set_xlim(-1, 2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    print(f'Save png to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


class ClipPlotSeriesWithBack:
    def __init__(self,
                 in_img_path,
                 in_mask_path,
                 in_back_img_path,
                 step_axial,
                 step_sagittal,
                 step_coronal,
                 num_clip,
                 vmin, vmax,
                 vmin_back, vmax_back,
                 unit_label):
        self._in_img_path = in_img_path
        self._in_img_name_no_ext = os.path.basename(self._in_img_path)
        self._in_mask_path = in_mask_path
        self._in_back_img_path = in_back_img_path
        self._step_axial = step_axial
        self._step_sagittal = step_sagittal
        self._step_coronal = step_coronal
        self._num_clip = num_clip
        self._vmin = vmin
        self._vmax = vmax
        self._vmin_back = vmin_back
        self._vmax_back = vmax_back
        self._sub_title_font_size = 70
        self._unit_label = unit_label

    def clip_plot_3_view(self, out_png_folder):
        in_img_obj = ScanWrapper(self._in_img_path)
        in_mask_obj = ScanWrapper(self._in_mask_path)

        in_img_data = in_img_obj.get_data()
        in_mask_data = in_mask_obj.get_data()

        masked_img_data = np.zeros(in_img_data.shape, dtype=float)
        masked_img_data.fill(np.nan)
        masked_img_data[in_mask_data == 1] = in_img_data[in_mask_data == 1]

    def clip_plot_img_only(self, out_png_folder):
        in_img_obj = ScanWrapper(self._in_img_path)
        in_img_data = in_img_obj.get_data()
        masked_img_data = None
        if self._in_mask_path is not None:
            in_mask_obj = ScanWrapper(self._in_mask_path)
            in_mask_data = in_mask_obj.get_data()

            masked_img_data = np.zeros(in_img_data.shape, dtype=float)
            masked_img_data.fill(np.nan)
            masked_img_data[in_mask_data == 1] = in_img_data[in_mask_data == 1]

        else:
            masked_img_data = in_img_data

        self._plot_view(
            self._num_clip,
            self._step_axial,
            masked_img_data,
            None,
            'axial',
            out_png_folder,
            1
        )

        self._plot_view(
            self._num_clip,
            self._step_sagittal,
            masked_img_data,
            None,
            'sagittal',
            out_png_folder,
            5.23438 / 2.28335
        )

        self._plot_view(
            self._num_clip,
            self._step_coronal,
            masked_img_data,
            None,
            'coronal',
            out_png_folder,
            5.23438 / 2.17388
        )

    def clip_plot(self, out_png_folder):
        in_img_obj = ScanWrapper(self._in_img_path)
        in_back_obj = ScanWrapper(self._in_back_img_path)

        in_img_data = in_img_obj.get_data()
        in_back_data = in_back_obj.get_data()

        masked_img_data = None
        masked_back_data = None

        if self._in_mask_path is not None:
            in_mask_obj = ScanWrapper(self._in_mask_path)
            in_mask_data = in_mask_obj.get_data()

            masked_img_data = np.zeros(in_img_data.shape, dtype=float)
            masked_img_data.fill(np.nan)
            masked_img_data[in_mask_data == 1] = in_img_data[in_mask_data == 1]

            masked_back_data = np.zeros(in_back_data.shape, dtype=float)
            masked_back_data.fill(np.nan)
            masked_back_data[in_mask_data == 1] = in_back_data[in_mask_data == 1]
        else:
            masked_img_data = in_img_data
            masked_back_data = in_back_data

        self._plot_view(
            self._num_clip,
            self._step_axial,
            masked_img_data,
            masked_back_data,
            'axial',
            out_png_folder,
            1
        )

        self._plot_view(
            self._num_clip,
            self._step_sagittal,
            masked_img_data,
            masked_back_data,
            'sagittal',
            out_png_folder,
            5.23438 / 2.28335
        )

        self._plot_view(
            self._num_clip,
            self._step_coronal,
            masked_img_data,
            masked_back_data,
            'coronal',
            out_png_folder,
            5.23438 / 2.17388
        )

    def _plot_view(self,
                   num_clip,
                   step_clip,
                   in_img_data,
                   in_back_data,
                   view_flag,
                   out_png_folder,
                   unit_ratio
                   ):
        for clip_idx in range(num_clip):
            fig, ax = plt.subplots()
            plt.axis('off')

            clip_off_set = (clip_idx - 2) * step_clip

            if in_back_data is not None:
                back_slice = self._clip_image(in_back_data, view_flag, clip_off_set)
                im_back = ax.imshow(
                    back_slice,
                    interpolation='none',
                    cmap='gray',
                    norm=colors.Normalize(vmin=self._vmin_back, vmax=self._vmax_back),
                    alpha=0.7
                )

            img_slice = self._clip_image(in_img_data, view_flag, clip_off_set)
            im = ax.imshow(
                img_slice,
                interpolation='none',
                cmap='jet',
                norm=colors.Normalize(vmin=self._vmin, vmax=self._vmax),
                alpha=0.5
            )

            ax.set_aspect(unit_ratio)

            if self._unit_label is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05/unit_ratio)

                cb = plt.colorbar(im, cax=cax)
                cb.set_label(self._unit_label)

            view_root = os.path.join(out_png_folder, f'{view_flag}')
            mkdir_p(view_root)
            out_png_path = os.path.join(view_root, f'{self._in_img_name_no_ext}_{clip_idx}.png')
            print(f'Save overlay png to {out_png_path}')
            plt.savefig(out_png_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    @staticmethod
    def _clip_image(image_data, clip_plane, offset=0):
        im_shape = image_data.shape
        clip = None
        if clip_plane == 'sagittal':
            clip = image_data[int(im_shape[0] / 2) - 1 + offset, :, :]
            clip = np.flip(clip, 0)
            clip = np.rot90(clip)
        elif clip_plane == 'coronal':
            clip = image_data[:, int(im_shape[1] / 2) - 1 + offset, :]
            clip = np.rot90(clip)
        elif clip_plane == 'axial':
            clip = image_data[:, :, int(im_shape[2] / 2) - 1 + offset]
            clip = np.rot90(clip)
        else:
            raise NotImplementedError

        return clip
