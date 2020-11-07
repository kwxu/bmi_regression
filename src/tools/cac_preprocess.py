import os
from tools.utils import get_logger, replace_nan
from tools.seg_lung import segment_a_lung, get_roi, get_cardiac_roi_mask
from tools.apply_mask import apply_mask
import nibabel as nib
import numpy as np


logger = get_logger('CAC preprocessing')


SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/../..'
c3d_path = os.path.join(SRC_ROOT, 'packages/c3d/c3d')
reg_resample_path = os.path.join(SRC_ROOT, 'packages/niftyreg/reg_resample')


class CACPreprocess:
    def __init__(self):
        pass

    @staticmethod
    def _step_resample(in_image_path, out_image_path, spacing):
        cmd_str = f'{c3d_path} {in_image_path} -resample-mm {spacing}x{spacing}x{spacing}mm -o {out_image_path}'
        logger.info('Resample image')
        logger.info(f'Run command: {cmd_str}')
        os.system(cmd_str)

    @staticmethod
    def _step_lung_mask(in_image_path, out_seg_path):
        logger.info(f'Get lung segmentation of {in_image_path}, save to {out_seg_path}')
        segment_a_lung(in_image_path, out_seg_path)

    @staticmethod
    def _step_get_lung_mask_bb(in_seg_path, out_bb_path):
        logger.info(f'Get the bb of {in_seg_path}, save to {out_bb_path}')
        get_roi(in_seg_path, out_bb_path)

    @staticmethod
    def _step_get_trimmed_bb(in_bb_path, out_trimmed_path):
        logger.info('Trim bounding box mask')
        cmd_str = f'{c3d_path} {in_bb_path} -trim 0vox -o {out_trimmed_path}'
        logger.info(f'Run command: {cmd_str}')
        os.system(cmd_str)

    @staticmethod
    def _step_roi_crop(in_ori_image_path, in_ref_path, out_roi_crop_path):
        logger.info('ROI crop')
        cmd_str = f'{reg_resample_path} -flo {in_ori_image_path} -ref {in_ref_path} -res {out_roi_crop_path} -inter 3'
        # cmd_str = f'{c3d_path} {in_ref_path} {in_ori_image_path} -mbb -o {out_roi_crop_path}'
        logger.info(f'Run command: {cmd_str}')
        os.system(cmd_str)

    @staticmethod
    def _step_roi_mask(in_lung_mask, out_roi_mask, kaggle_format=True):
        logger.info(f'Create CAC ROI mask using {in_lung_mask}, save to {out_roi_mask}')
        get_cardiac_roi_mask(in_lung_mask, out_roi_mask, kaggle_format)

    @staticmethod
    def _step_apply_mask(in_ori_image_path, mask_path, out_apply_mask_path, ambient):
        logger.info(f'Apply mask {mask_path} to {in_ori_image_path} with ambient {ambient}')
        logger.info(f'Save to {out_apply_mask_path}')
        apply_mask(in_ori_image_path, mask_path, ambient, out_apply_mask_path)

    @staticmethod
    def _step_apply_window(in_ori_image_path, window_lower, window_upper, out_image_path):
        logger.info(f'Apply window [{window_lower}, {window_upper}]')
        cmd_str = f'{c3d_path} {in_ori_image_path} -clip {window_lower} {window_upper} -o {out_image_path}'
        logger.info(f'Run command: {cmd_str}')
        os.system(cmd_str)

    @staticmethod
    def _step_calcium_mask(in_ori_image_path, thres_val, out_mask_path):
        logger.info(f'Get calcium mask of {in_ori_image_path} with thres_val {thres_val}, save to {out_mask_path}')
        cmd_str = f'{c3d_path} {in_ori_image_path} -threshold {thres_val} inf 1 0 -o {out_mask_path}'
        logger.info(f'Run command: {cmd_str}')
        os.system(cmd_str)

    @staticmethod
    def _step_resample_final(in_ori_image_path, out_image_path, interp_order=3):
        cmd_str = f'{c3d_path} -int {interp_order} {in_ori_image_path} -resample 192x128x64 -o {out_image_path}'
        logger.info(f'Run command: {cmd_str}')
        os.system(cmd_str)

    @staticmethod
    def _step_replace_nan(in_image_path, out_image_path, impute_value=0):
        replace_nan(in_image_path, out_image_path, impute_value)

    @staticmethod
    def _step_merge_jac_calcium(in_jac_path, in_calcium_path, out_calcium_w_vol_path):
        logger.info(f'Merge jac {in_jac_path} and calcium {in_calcium_path}')
        jac_data = nib.load(in_jac_path)
        jac_img = jac_data.get_data()
        cal_img = nib.load(in_calcium_path).get_data()
        cal_img[cal_img < 0.5] = 0

        jac_no_log_img = np.power(10, jac_img)
        out_cal_w_vol = np.multiply(cal_img, jac_no_log_img)
        logger.info(f'Save cal_w_vol to {out_calcium_w_vol_path}')
        out_obj = nib.Nifti1Image(out_cal_w_vol, affine=jac_data.affine, header=jac_data.header)
        nib.save(out_obj, out_calcium_w_vol_path)

    @staticmethod
    def _step_unlog(in_jac_path, out_unlog_jac_path):
        logger.info(f'Unlog {in_jac_path}, save to {out_unlog_jac_path}')
        jac_data = nib.load(in_jac_path)
        jac_img = jac_data.get_data()
        print(f'{np.min(jac_img)}, {np.max(jac_img)}')
        jac_img = np.clip(jac_img, -2, 2)
        print(f'{np.min(jac_img)}, {np.max(jac_img)}')
        # jac_img = np.power(np.pi, jac_img)
        jac_img = np.exp(jac_img)
        print(f'{np.min(jac_img)}, {np.max(jac_img)}')
        out_jac_obj = nib.Nifti1Image(jac_img, affine=jac_data.affine, header=jac_data.header)
        nib.save(out_jac_obj, out_unlog_jac_path)