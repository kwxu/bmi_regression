import os
import numpy as np
import nibabel as nib
import random
from scipy import ndarray
from skimage import transform, util


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-15, 15)
    return transform.rotate(image_array, random_degree, preserve_range=True)


def random_rotation_twoarrays(image_array,mask_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-15, 15)
    rot_img = transform.rotate(image_array, random_degree, preserve_range=True)
    rot_mask = transform.rotate(mask_array, random_degree, preserve_range=True)
    return rot_img,rot_mask


def random_rotation_N_arrays(multi_channel_img):
    random_degree = random.uniform(-15, 15)
    rot_img = multi_channel_img.copy()
    for idx_channel in range(multi_channel_img.shape[0]):
        rot_img[idx_channel] = transform.rotate(multi_channel_img[idx_channel], random_degree, preserve_range=True)

    return rot_img

'''
def random_noise(image_array: ndarray):
    # add random noise to the image
    return util.random_noise(image_array)
'''

#in ZXY representation
def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:,:,::-1]

#in ZXY representation
def vertical_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1,:]

'''
#augmentation
if(random.uniform(0, 1)>0.7):
rot1,rot2=random.sample(set([0,1,2]), 2)
reshaped=np.rot90(reshaped,axes=(rot1,rot2))
'''


def random_translation(image_array, limit):
    seed = np.random.randint(limit, size=6)
    choice = np.random.choice(2, 3)

    rt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        rt = util.pad(rt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    return rt


def random_translation_twoarrays(image_array, mask_array, limit):
    seed = np.random.randint(limit, size=6)
    choice = np.random.choice(2, 3)

    rt = util.crop(copy=True, ar=image_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        rt = util.pad(rt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    mt = util.crop(copy=True, ar=mask_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        mt = util.pad(mt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        mt = util.pad(mt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        mt = util.pad(mt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        mt = util.pad(mt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    return rt, mt


def _translate_one_array(one_array, seed, choice):
    rt = util.crop(copy=True, ar=one_array, crop_width=((seed[0], seed[1]), (seed[2], seed[3]), (seed[4], seed[5])))

    if (choice[0] == 0):
        rt = util.pad(rt, ((seed[0] + seed[1], 0), (0, 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, seed[0] + seed[1]), (0, 0), (0, 0)), 'constant')

    if (choice[1] == 0):
        rt = util.pad(rt, ((0, 0), (seed[2] + seed[3], 0), (0, 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, seed[2] + seed[3]), (0, 0)), 'constant')

    if (choice[2] == 0):
        rt = util.pad(rt, ((0, 0), (0, 0), (seed[4] + seed[5], 0)), 'constant')
    else:
        rt = util.pad(rt, ((0, 0), (0, 0), (0, seed[4] + seed[5])), 'constant')

    return rt

def random_translation_N_array(multi_channel_array, limit):
    seed = np.random.randint(limit, size=6)
    choice = np.random.choice(2, 3)

    new_array = multi_channel_array.copy()

    for idx_channel in range(multi_channel_array.shape[0]):
        rt = _translate_one_array(multi_channel_array[idx_channel], seed, choice)
        new_array[idx_channel] = rt

    return new_array