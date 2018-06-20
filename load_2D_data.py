'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for loading training, validation, and testing data into the models.
It is specifically designed to handle 3D single-channel medical data.
Modifications will be needed to train/test on normal 3-channel images.

Enhancement:
    0. Porting to Python version 3.6
    1. Add image_resize2square to accept any size of images and change to 512 X 512 resolutions.
    2. 
'''

from __future__ import print_function

from os.path import join, basename
from os import mkdir
import csv
import numpy as np
from numpy.random import rand, shuffle
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm #Progress bar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

plt.ioff()
import threading
from load_data import augmentImages, image_resize2square

# 
# def flip_axis(x, axis):
#     x = np.asarray(x).# 
# def flip_axis(x, axis):
#     x = np.asarray(x).swapaxes(axis, 0)
#     x = x[::-1, ...]
#     x = x.swapaxes(0, axis)IMAGE_SIZE = 512IMAGE_SIZE = 512
#     return xswapaxes(axis, 0)
#     x = x[::-1, ...]
#     x = x.swapaxes(0, axis)
#     return x

debug = 0
IMAGE_SIZE = 512
COCO_BACKGROUND = (61, 1, 84, 255)
MASK_BACKGROUND = (0,0,0,0)      

def change_background_color(img, original_color, new_color):
    '''
    Convert mask color of 4 channels png image to new color 
    '''
    
    r1, g1, b1, a1 = original_color[0], original_color[1], original_color[2], original_color[3]  # Original value
    r2, g2, b2, a2 = new_color[0], new_color[1], new_color[2], new_color[3] # Value that we want to replace it with

    red, green, blue, alpha = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
    mask = (red == r1) & (green == g1) & (blue == b1) & (alpha == a1)
    img[:,:,:4][mask] = [r2, g2, b2, a2]
    return img

def convert_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False):
    fname = img_name[:-4]
    numpy_path = join(root_path, 'np_files')
    img_path = join(root_path, 'imgs')
    mask_path = join(root_path, 'masks')
    fig_path = join(root_path, 'figs')
    try:
        mkdir(numpy_path)
    except:
        pass
    try:
        mkdir(fig_path)
    except:
        pass

#         ct_min = -1024 # No need for 2d image
#         ct_max = 3072

    if not overwrite:
        try:
            with np.load(join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            pass

    try:
        itk_img = sitk.ReadImage(join(img_path, img_name))
        img = sitk.GetArrayFromImage(itk_img)
        img = image_resize2square(img, IMAGE_SIZE)
        img = img[:,:,:3] # Only get RGB channels. Remove alpha channel.

        if not no_masks:
            itk_mask = sitk.ReadImage(join(mask_path, img_name))
            mask = sitk.GetArrayFromImage(itk_mask)
            mask = image_resize2square(mask, IMAGE_SIZE)
            
            mask = change_background_color(mask, COCO_BACKGROUND, MASK_BACKGROUND) 
            mask = mask[:,:,:3] # Only get RGB channels. Remove alpha channel.      

            mask[mask >= 1] = 1 # Person
            mask[mask != 1] = 0 # Non Person / Background
            mask = mask.astype(np.uint8)

        if not no_masks:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img, mask=mask)
        else:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask
        else:
            return img

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)
# 
# def flip_axis(x, axis):
#     x = np.asarray(x).swapaxes(axis, 0)
#     x = x[::-1, ...]
#     x = x.swapaxes(0, axis)
#     return x


def get_slice(image_data):
    return image_data[2]

''' Make the generators threadsafe in case of multiple threads '''
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # python 3
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generate_train_batches(root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):
    # Create placeholders for training
    # (img_shape[1], img_shape[2], args.slices)
    print('==>2d_generate_train_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
                print('==>path_to_np=%s'%(path_to_np))
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                train_img, train_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1: # Only one slices. code can be removed.
                np.random.seed(None)
                subSampAmt = int(rand(1)*(train_img.shape[2]*0.05))
            # We don't need indicies in 2D image.
            indicies = np.arange(0, train_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = train_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :] = train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = train_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :, 0] = train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if aug_data:
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)
                    if debug:
                        if img_batch.ndim == 4:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                        elif img_batch.ndim == 5:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
                        plt.savefig(join(root_path, 'logs', 'ex_train.png'), format='png', bbox_inches='tight')
                        plt.close()
                    if net.find('caps') != -1: # if the network is capsule/segcaps structure
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if aug_data:
                img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],
                                                                              mask_batch[:count,...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_val_batches(root_path, val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                         stride=1, downSampAmt=1, shuff=1):
    print('==>2d_generate_val_batches')
    # Create placeholders for validation
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for i, scan_name in enumerate(val_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
                with np.load(path_to_np) as data:
                    val_img = data['img']
                    val_mask = data['mask']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                val_img, val_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(val_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')

            indicies = np.arange(0, val_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = val_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :] = val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = val_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :, 0] = val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                else:
                    print('Error this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_test_batches(root_path, test_list, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    # Create placeholders for testing
    print('load_2D_data.generate_test_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    count = 0
    print('load_2D_data.generate_test_batches: test_list=%s'%(test_list))
    for i, scan_name in enumerate(test_list):
        try:
            scan_name = scan_name[0]
            path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
            with np.load(path_to_np) as data:
                test_img = data['img']
        except:
            print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
            test_img = convert_data_to_numpy(root_path, scan_name, no_masks=True)
            if np.array_equal(test_img,np.zeros(1)):
                continue
            else:
                print('\nFinished making npz file.')

        indicies = np.arange(0, test_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
        for j in indicies:
            if img_batch.ndim == 4:
                img_batch[count, :, :, :] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, :, :, :, 0] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            else:
                print('Error this function currently only supports 2D and 3D data.')
                exit(0)

            count += 1
            if count % batchSize == 0:
                count = 0
                yield (img_batch)

    if count != 0:
        yield (img_batch[:count,:,:,:])
    