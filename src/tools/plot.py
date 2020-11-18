import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


def plot_prediction_scatter(pred_df, out_png):
    pred = pred_df['pred'].to_numpy()
    label = pred_df['target'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(label, pred)



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
