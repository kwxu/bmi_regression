import os
import numpy as np
from torch.utils import data
import nibabel as nib
import random
from random import randint
from tools.image_manipulation import random_rotation, random_rotation_twoarrays, random_translation, random_translation_twoarrays
from tools.image_manipulation import random_rotation_N_arrays, random_translation_N_array, apply_valid_region_mask


# output_x = 168
# output_y = 168
# output_z = 64


class pytorch_loader_clss3D(data.Dataset):
    def __init__(
            self,
            subdict,
            config,
            suppress_data_augmentation=False
    ):
        self.subdict = subdict
        self.img_names = subdict['img_names']
        self.img_subs = subdict['img_subs']
        self.img_files = subdict['img_files']
        self.gt_val = subdict['gt_val']
        res = config['res']
        imsize = config['imsize']
        self.output_x = res[0]
        self.output_y = res[1]
        self.output_z = res[2]
        self.img_x = imsize[0]
        self.img_y = imsize[1]
        self.img_z = imsize[2]
        self.data_augmentation = config['data_augmentation']
        self.suppress_data_augmentation = suppress_data_augmentation

        self.add_jac_map = config['add_jacobian_map']
        if self.add_jac_map:
            self.jacobian_maps = subdict['jacobian_maps']

        self.add_valid_mask = config['add_valid_mask_map']
        if self.add_valid_mask | config['apply_random_valid_mask']:
            self.valid_masks = subdict['valid_masks']

        self.add_d_index_mask = config['add_d_index_map']
        if self.add_d_index_mask:
            self.d_index_masks = subdict['d_index_maps']

        self.add_jac_elem_maps = config['add_jac_elem_maps']
        if self.add_jac_elem_maps:
            self.jac_elem_maps = []
            for idx_elem in range(9):
                self.jac_elem_maps.append(subdict[f'jac_elem_{idx_elem}_map'])

        self.config = config

        self.if_print_debug = False

    @staticmethod
    def _add_channel(x, img):
        new_x = None
        if x is None:
            new_x = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=float)
            new_x[0] = img
        else:
            num_channel = x.shape[0]
            new_x = np.zeros((num_channel + 1, x.shape[1], x.shape[2], x.shape[3]))
            new_x[:num_channel, :, :, :] = x
            new_x[num_channel, :, :, :] = img

        return new_x

    def __getitem__(self, index):
        # sub_name = self.img_subs[index]
        file_name = self.img_names[index]

        x = None

        def add_channel(data, in_file, cfg_max_str, cfg_min_str):
            img_obj = nib.load(in_file)
            img = None
            try:
                img = img_obj.get_data()
            except:
                print('******************** %s\n'%in_file)

            img = np.transpose(img, (2, 0, 1))
            if cfg_max_str is not None:
                img_max = self.config[cfg_max_str]
                img_min = self.config[cfg_min_str]
                img = np.clip(img, img_min, img_max)
                img = (img - img_min) / (img_max - img_min)
            img = img * 255.0
            data = pytorch_loader_clss3D._add_channel(data, img)

            return data

        if self.config['add_intensity_map']:
            in_file = self.img_files[index]
            x = add_channel(x, in_file, 'img_max', 'img_min')

        if self.add_jac_map:
            in_file = self.jacobian_maps[index]
            x = add_channel(x, in_file, 'jac_max', 'jac_min')

        if self.add_d_index_mask:
            in_file = self.d_index_masks[index]
            x = add_channel(x, in_file, 'd_index_max', 'd_index_min')

        if self.add_valid_mask:
            in_file = self.valid_masks[index]
            x = add_channel(x, in_file, None, None)

        if self.add_jac_elem_maps:
            for idx_elem in range(9):
                in_file = self.jac_elem_maps[idx_elem][index]
                x = add_channel(x, in_file, 'jac_elem_max', 'jac_elem_min')

        augmentation_pad_val = self.config['ambient_val']

        if self.config['apply_valid_mask']:
            in_file = self.valid_masks[index]
            img_obj = nib.load(in_file)
            mask_data = img_obj.get_data()
            mask_data = np.transpose(mask_data, (2, 0, 1))
            # ambient_val = 0
            # augmentation_pad_val = ambient_val
            x = apply_valid_region_mask(x, mask_data, augmentation_pad_val)

        # Apply random mask on training phase.
        if self.config['apply_random_valid_mask']:
            mask_idx = index
            if not self.suppress_data_augmentation:
                if self.if_print_debug:
                    print(f'Get random index for valid mask.')
                mask_idx = randint(0, len(self.valid_masks) - 1)
            in_file = self.valid_masks[mask_idx]
            if self.if_print_debug:
                print(f'Session index: {index}, {file_name}')
                print(f'Apply valid mask index: {mask_idx}, {os.path.basename(in_file)}')
            img_obj = nib.load(in_file)
            mask_data = img_obj.get_data()
            mask_data = np.transpose(mask_data, (2, 0, 1))
            x = apply_valid_region_mask(x, mask_data, augmentation_pad_val)

        if self.data_augmentation & (not self.suppress_data_augmentation):
            if self.if_print_debug:
                print(f'Enter data augmentation')
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.5):
                x = random_rotation_N_arrays(x, pad_val=augmentation_pad_val)
                x = random_translation_N_array(x, 5, pad_val=augmentation_pad_val)

        x = x.astype('float32')

        if self.if_print_debug & self.suppress_data_augmentation:
            out_folder = '/nfs/masi/xuk9/SPORE/CAC_class/debug'
            out_file_path = os.path.join(out_folder, file_name)
            print(x.shape)
            img_obj = nib.Nifti1Image(x[0], affine=None)
            print(f'Save network input to {out_file_path}')
            nib.save(img_obj, out_file_path)

        y = np.array([self.gt_val[index]])

        return x, y, file_name

    def __len__(self):
        return len(self.img_subs)
