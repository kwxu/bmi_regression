import os
from tools.utils import *
import argparse
import numpy as np
import nibabel as nib


def apply_mask(ori, mask, ambient, out):
    ambient_val = None
    ambient_val_str = ambient
    if ambient_val_str is 'nan':
        ambient_val = np.nan
    else:
        ambient_val = float(ambient)

    ori_img_obj = nib.load(ori)
    mask_img_obj = nib.load(mask)

    ori_img = ori_img_obj.get_data()
    # ori_img = np.nan_to_num(ori_img, nan=ambient)
    mask_img = mask_img_obj.get_data()

    mask_img[mask_img < 0.5] = 0

    new_img_data = np.full(ori_img.shape, ambient_val)
    np.copyto(new_img_data, ori_img, where=mask_img > 0)

    masked_img_obj = nib.Nifti1Image(new_img_data, affine=ori_img_obj.affine, header=ori_img_obj.header)
    nib.save(masked_img_obj, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori', type=str)
    parser.add_argument('--mask', type=str)
    parser.add_argument('--ambient', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    apply_mask(args.ori, args.mask, args.ambient, args.out)


if __name__ == '__main__':
    main()
