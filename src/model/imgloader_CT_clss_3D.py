import os
import numpy as np
from torch.utils import data
import nibabel as nib
import random
from tools.image_manipulation import random_rotation, random_rotation_twoarrays, random_translation, random_translation_twoarrays
from tools.image_manipulation import random_rotation_N_arrays, random_translation_N_array
# output_x = 168
# output_y = 168
# output_z = 64



class pytorch_loader_clss3D(data.Dataset):
    def __init__(
            self,
            subdict,
            config):
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

        self.add_jac_map = config['add_jacobian_map']
        if self.add_jac_map:
            self.jacobian_maps = subdict['jacobian_maps']

        self.config = config

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
        if self.config['add_intensity_map']:
            img_file = self.img_files[index]
            img_3d = nib.load(img_file)
            img = None
            try:
                img = img_3d.get_data()
            except:
                print('******************** %s\n'%img_file)

            img = np.transpose(img, (2, 0, 1))
            # img = (img - img.min()) / (img.max() - img.min())
            img_max = self.config['img_max']
            img_min = self.config['img_min']
            img = np.clip(img, img_min, img_max)
            img = (img - img_min) / (img_max - img_min)
            img = img * 255.0
            x = pytorch_loader_clss3D._add_channel(x, img)

        if self.add_jac_map:
            jac_map_file = self.jacobian_maps[index]
            jac_map = nib.load(jac_map_file)
            jac_img = None
            try:
                jac_img = jac_map.get_data()
            except:
                print(f'*******************{jac_map_file} \n')

            jac_img = np.transpose(jac_img, (2, 0, 1))

            jac_max = self.config['jac_max']
            jac_min = self.config['jac_min']
            jac_img = np.clip(jac_img, jac_min, jac_max)
            jac_img = (jac_img - jac_min) / (jac_max - jac_min)
            jac_img = jac_img * 255.0
            x = pytorch_loader_clss3D._add_channel(x, jac_img)

        if self.data_augmentation:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.5):
                x = random_rotation_N_arrays(x)
                x = random_translation_N_array(x, 5)
                # img, calcium_img = random_rotation_twoarrays(img, calcium_img)
                # img, calcium_img = random_translation_twoarrays(img, calcium_img, 5)

        x = x.astype('float32')
        y = np.array([self.gt_val[index]])

        return x, y, file_name

    def __len__(self):
        return len(self.img_subs)
