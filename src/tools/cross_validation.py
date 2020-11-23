from tools.utils import get_logger
from tools.clinical import ClinicalDataReaderSPORE
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


logger = get_logger('Cross-validation utilities.')


def show_subject_label_fold_statistics(
        num_fold,
        subject_train_idx_list_array,
        scan_train_idx_list_array,
        subject_label,
        set_flag='Train'):
    fold_train_subject_label_statics_dict_list = []
    for idx_fold in range(num_fold):
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        train_label = np.array([subject_label[idx] for idx in subject_train_idx_list])

        train_unique, train_counts = np.unique(train_label, return_counts=True)
        train_dict = dict(zip(train_unique, train_counts))
        fold_train_subject_label_statics_dict_list.append(train_dict)

    logger.info(f'Sizes of each fold:')
    logger.info(f'# {set_flag} (subject): {[len(train_subject_list) for train_subject_list in subject_train_idx_list_array]}')
    logger.info(f'# {set_flag} (scan): {[len(train_list) for train_list in scan_train_idx_list_array]}')
    for idx_fold in range(num_fold):
        logger.info(f'# {set_flag} label (subject, fold-{idx_fold}): {fold_train_subject_label_statics_dict_list[idx_fold]}')


def get_idx_list_array_bmi_session_level_split(bmi_list, num_fold):
    """
    Session level split (different from split at subject level)
    This is design for BMI.
    The train valid test fold rotationing is following the same rule as previous.
    :param bmi_list:
    :param num_fold:
    :return:
    """
    num_stratify = 5
    percentile_step = 100 / num_stratify
    split_loc = np.zeros((num_stratify + 1,), dtype=float)
    split_loc[0] = np.min(bmi_list) - 1
    split_loc[-1] = np.max(bmi_list) + 1

    for idx_percentile in range(1, num_stratify):
        percentile_split_loc = np.percentile(bmi_list, idx_percentile * percentile_step)
        split_loc[idx_percentile] = percentile_split_loc

    label_vec = np.zeros(bmi_list.shape, dtype=int)
    for idx_percentile in range(num_stratify):
        label_vec[(bmi_list > split_loc[idx_percentile]) & (bmi_list <= split_loc[idx_percentile + 1])] = idx_percentile

    skf = StratifiedKFold(n_splits=num_fold, random_state=0, shuffle=True)
    scan_fold_idx_list_array = []
    # Training group is ignored
    for train_idx_list, test_idx_list in skf.split(bmi_list, label_vec):
        scan_fold_idx_list_array.append(test_idx_list)

    scan_train_idx_list_array = []
    scan_validate_idx_list_array = []
    scan_test_idx_list_array = []
    for idx_fold in range(num_fold):
        cur_idx_fold = idx_fold
        scan_train_idx_list = []
        for idx_train_fold in range(num_fold - 2):
            scan_train_idx_list.append(scan_fold_idx_list_array[cur_idx_fold])
            cur_idx_fold = (cur_idx_fold + 1) % num_fold
        scan_train_idx_list = np.concatenate(scan_train_idx_list)
        scan_train_idx_list_array.append(scan_train_idx_list)
        scan_validate_idx_list_array.append(scan_fold_idx_list_array[cur_idx_fold])
        cur_idx_fold = (cur_idx_fold + 1) % num_fold
        scan_test_idx_list_array.append(scan_fold_idx_list_array[cur_idx_fold])

    return scan_train_idx_list_array, scan_validate_idx_list_array, scan_test_idx_list_array


def get_idx_list_array_n_fold_regression_bl(file_name_list, num_fold):
    subject_id_full = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
                       for file_name in file_name_list]
    subject_id_unique = list(set(subject_id_full))
    kf = KFold(n_splits=num_fold, random_state=0)

    subject_fold_idx_list_array = []
    for train_idx_list, test_idx_list in kf.split(subject_id_unique):
        subject_fold_idx_list_array.append(test_idx_list)

    subject_train_idx_list_array = []
    subject_validate_idx_list_array = []
    subject_test_idx_list_array = []
    for idx_fold in range(num_fold):
        cur_idx_fold = idx_fold
        fold_subject_train_idx_list = []
        for idx_train_fold in range(num_fold - 2):
            fold_subject_train_idx_list.append(subject_fold_idx_list_array[cur_idx_fold])
            cur_idx_fold = (cur_idx_fold + 1) % num_fold
        fold_subject_train_idx_list = np.concatenate(fold_subject_train_idx_list)
        subject_train_idx_list_array.append(fold_subject_train_idx_list)
        subject_validate_idx_list_array.append(subject_fold_idx_list_array[cur_idx_fold])
        cur_idx_fold = (cur_idx_fold + 1) % num_fold
        subject_test_idx_list_array.append(subject_fold_idx_list_array[cur_idx_fold])

    scan_train_idx_list_array = []
    scan_validate_idx_list_array = []
    scan_test_idx_list_array = []
    for idx_fold in range(num_fold):
        scan_train_idx_list = []
        scan_validate_idx_list = []
        scan_test_idx_list = []
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        subject_validate_idx_list = subject_validate_idx_list_array[idx_fold]
        subject_test_idx_list = subject_test_idx_list_array[idx_fold]

        for idx_subject in subject_train_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_train_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_train_idx_list += subject_scan_train_idx_list

        for idx_subject in subject_validate_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_validate_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_validate_idx_list += subject_scan_validate_idx_list

        for idx_subject in subject_test_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_test_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_test_idx_list += subject_scan_test_idx_list

        scan_train_idx_list_array.append(scan_train_idx_list)
        scan_validate_idx_list_array.append(scan_validate_idx_list)
        scan_test_idx_list_array.append(scan_test_idx_list)

    return scan_train_idx_list_array, scan_validate_idx_list_array, scan_test_idx_list_array


def get_idx_list_array_n_fold_cross_validation_bl(file_name_list, label_list, num_fold):
    """
    Get the n-folder split at subject level (scans of the same subject always go into one fold)
    :param file_name_list: file name list of scans, with .nii.gz
    :param num_fold: number of folds
    :return:
    """
    scan_label = label_list
    subject_id_full = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
                       for file_name in file_name_list]
    subject_id_unique = list(set(subject_id_full))
    subject_label = [label_list[subject_id_full.index(subject_id)]
                     for subject_id in subject_id_unique]

    skf = StratifiedKFold(n_splits=num_fold, random_state=0)

    subject_fold_idx_list_array = []
    for train_idx_list, test_idx_list in skf.split(subject_id_unique, subject_label):
        subject_fold_idx_list_array.append(test_idx_list)

    subject_train_idx_list_array = []
    subject_validate_idx_list_array = []
    subject_test_idx_list_array = []
    for idx_fold in range(num_fold):
        cur_idx_fold = idx_fold
        fold_subject_train_idx_list = []
        for idx_train_fold in range(num_fold - 2):
            fold_subject_train_idx_list.append(subject_fold_idx_list_array[cur_idx_fold])
            cur_idx_fold = (cur_idx_fold + 1) % num_fold
        fold_subject_train_idx_list = np.concatenate(fold_subject_train_idx_list)
        subject_train_idx_list_array.append(fold_subject_train_idx_list)
        subject_validate_idx_list_array.append(subject_fold_idx_list_array[cur_idx_fold])
        cur_idx_fold = (cur_idx_fold + 1) % num_fold
        subject_test_idx_list_array.append(subject_fold_idx_list_array[cur_idx_fold])

    scan_train_idx_list_array = []
    scan_validate_idx_list_array = []
    scan_test_idx_list_array = []
    for idx_fold in range(num_fold):
        scan_train_idx_list = []
        scan_validate_idx_list = []
        scan_test_idx_list = []
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        subject_validate_idx_list = subject_validate_idx_list_array[idx_fold]
        subject_test_idx_list = subject_test_idx_list_array[idx_fold]

        for idx_subject in subject_train_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_train_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_train_idx_list += subject_scan_train_idx_list

        for idx_subject in subject_validate_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_validate_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_validate_idx_list += subject_scan_validate_idx_list

        for idx_subject in subject_test_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_test_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_test_idx_list += subject_scan_test_idx_list

        scan_train_idx_list_array.append(scan_train_idx_list)
        scan_validate_idx_list_array.append(scan_validate_idx_list)
        scan_test_idx_list_array.append(scan_test_idx_list)

    show_subject_label_fold_statistics(
        num_fold,
        subject_train_idx_list_array,
        scan_train_idx_list_array,
        subject_label,
        set_flag='Train'
    )
    show_subject_label_fold_statistics(
        num_fold,
        subject_validate_idx_list_array,
        scan_validate_idx_list_array,
        subject_label,
        set_flag='Validate'
    )
    show_subject_label_fold_statistics(
        num_fold,
        subject_test_idx_list_array,
        scan_test_idx_list_array,
        subject_label,
        set_flag='Test'
    )

    return scan_train_idx_list_array, scan_validate_idx_list_array, scan_test_idx_list_array


def get_idx_list_array_n_fold_cross_validation(file_name_list, label_list, num_fold):
    """
    Get the n-folder split at subject level (scans of the same subject always go into one fold)
    :param file_name_list: file name list of scans, with .nii.gz
    :param num_fold: number of folds
    :return:
    """
    scan_label = label_list
    subject_id_full = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
                       for file_name in file_name_list]
    subject_id_unique = list(set(subject_id_full))
    subject_label = [label_list[subject_id_full.index(subject_id)]
                     for subject_id in subject_id_unique]

    skf = StratifiedKFold(n_splits=num_fold, random_state=0)
    # skf = KFold(n_splits=num_fold, random_state=0)
    # logger.info(f'Split data set into {skf.get_n_splits()} folds.')
    # logger.info(f'Number of scans: {len(file_name_list)}')
    # logger.info(f'Number of subjects: {len(subject_id_unique)}')

    subject_train_idx_list_array = []
    subject_test_idx_list_array = []
    for train_idx_list, test_idx_list in skf.split(subject_id_unique, subject_label):
        subject_train_idx_list_array.append(train_idx_list)
        subject_test_idx_list_array.append(test_idx_list)

    # for train_idx_list, test_idx_list in skf.split(subject_id_unique):
    #     subject_train_idx_list_array.append(train_idx_list)
    #     subject_test_idx_list_array.append(test_idx_list)

    scan_train_idx_list_array = []
    scan_test_idx_list_array = []
    for idx_fold in range(num_fold):
        scan_train_idx_list = []
        scan_test_idx_list = []
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        subject_test_idx_list = subject_test_idx_list_array[idx_fold]

        for idx_subject in subject_train_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_train_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_train_idx_list += subject_scan_train_idx_list

        for idx_subject in subject_test_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_test_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_test_idx_list += subject_scan_test_idx_list

        scan_train_idx_list_array.append(scan_train_idx_list)
        scan_test_idx_list_array.append(scan_test_idx_list)

    num_pos_scan_train_fold_array = []
    num_pos_scan_test_fold_array = []
    num_pos_subject_train_fold_array = []
    num_pos_subject_test_fold_array = []

    fold_train_subject_label_statics_dict_list = []
    fold_test_subject_label_statics_dict_list = []
    for idx_fold in range(num_fold):
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        subject_test_idx_list = subject_test_idx_list_array[idx_fold]
        train_label = np.array([subject_label[idx] for idx in subject_train_idx_list])
        test_label = np.array([subject_label[idx] for idx in subject_test_idx_list])

        train_unique, train_counts = np.unique(train_label, return_counts=True)
        train_dict = dict(zip(train_unique, train_counts))
        fold_train_subject_label_statics_dict_list.append(train_dict)

        test_unique, test_counts = np.unique(test_label, return_counts=True)
        test_dict = dict(zip(test_unique, test_counts))
        fold_test_subject_label_statics_dict_list.append(test_dict)

    logger.info(f'Sizes of each fold:')
    logger.info(f'# Train (subject): {[len(train_subject_list) for train_subject_list in subject_train_idx_list_array]}')
    logger.info(f'# Test (subject): {[len(test_subject_list) for test_subject_list in subject_test_idx_list_array]}')
    logger.info(f'# Train (scan): {[len(train_list) for train_list in scan_train_idx_list_array]}')
    logger.info(f'# Test (scan): {[len(test_list) for test_list in scan_test_idx_list_array]}')
    for idx_fold in range(num_fold):
        logger.info(f'# Train label (subject, fold-{idx_fold}): {fold_train_subject_label_statics_dict_list[idx_fold]}')
    for idx_fold in range(num_fold):
        logger.info(f'# Test label (subject, fold-{idx_fold}): {fold_test_subject_label_statics_dict_list[idx_fold]}')

    return scan_train_idx_list_array, scan_test_idx_list_array

