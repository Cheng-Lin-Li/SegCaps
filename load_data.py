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

# import threading
from os.path import join, basename
from os import mkdir
from glob import glob
import csv
from sklearn.model_selection import KFold
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

from keras.preprocessing.image import *

from custom_data_aug import elastic_transform, salt_pepper_noise

debug = 0

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

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def image_resize2square(image, desired_size = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    old_size = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if desired_size is None or (old_size[0]==desired_size and old_size[1]==desired_size):
        return image

    # calculate the ratio of the height and construct the
    # dimensions
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    resized = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # return the resized image
    return new_image

def load_data(root, split):
    # Load the training and testing lists
    with open(join(root, 'split_lists', 'train_split_' + str(split) + '.csv'), 'r') as f:
        reader = csv.reader(f)
        training_list = list(reader)

    with open(join(root, 'split_lists', 'test_split_' + str(split) + '.csv'), 'r') as f:
        reader = csv.reader(f)
        testing_list = list(reader)

    new_training_list, validation_list = train_test_split(training_list, test_size=0.1, random_state=7)

    return new_training_list, validation_list, testing_list

def compute_class_weights(root, train_data_list):
    '''
        We want to weight the the positive pixels by the ratio of negative to positive.
        Three scenarios:
            1. Equal classes. neg/pos ~ 1. Standard binary cross-entropy
            2. Many more negative examples. The network will learn to always output negative. In this way we want to
               increase the punishment for getting a positive wrong that way it will want to put positive more
            3. Many more positive examples. We weight the positive value less so that negatives have a chance.
    '''
    pos = 0.0
    neg = 0.0
    for img_name in tqdm(train_data_list):
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(root, 'masks', img_name[0])))
        for slic in img:
            if not np.any(slic):
                continue
            else:
                p = np.count_nonzero(slic)
                pos += p
                neg += (slic.size - p)

    return neg/pos

def load_class_weights(root, split):
    class_weight_filename = join(root, 'split_lists', 'train_split_' + str(split) + '_class_weights.npy')
    try:
        return np.load(class_weight_filename)
    except:
        print('Class weight file {} not found.\nComputing class weights now. This may take '
              'some time.'.format(class_weight_filename))
        train_data_list, _, _ = load_data(root, str(split))
        value = compute_class_weights(root, train_data_list)
        np.save(class_weight_filename,value)
        print('Finished computing class weights. This value has been saved for this training split.')
        return value


def split_data(root_path, num_splits=4):
    mask_list = []
    for ext in ('*.mhd', '*.hdr', '*.nii', '*.png'): #add png file support
        mask_list.extend(sorted(glob(join(root_path,'imgs',ext)))) # check imgs instead of masks

    assert len(mask_list) != 0, 'Unable to find any files in {}'.format(join(root_path,'imgs'))

    outdir = join(root_path,'split_lists')
    try:
        mkdir(outdir)
    except:
        pass

    kf = KFold(n_splits=num_splits)
    n = 0
    for train_index, test_index in kf.split(mask_list):
        with open(join(outdir,'train_split_' + str(n) + '.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in train_index:
                print('basename=%s'%([basename(mask_list[i])]))
                writer.writerow([basename(mask_list[i])])
        with open(join(outdir,'test_split_' + str(n) + '.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in test_index:
                writer.writerow([basename(mask_list[i])])
        n += 1

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def augmentImages(batch_of_images, batch_of_masks):
    for i in range(len(batch_of_images)):
        img_and_mask = np.concatenate((batch_of_images[i, ...], batch_of_masks[i,...]), axis=2)
        if img_and_mask.ndim == 4: # This assumes single channel data. For multi-channel you'll need
            # change this to put all channel in slices channel
            orig_shape = img_and_mask.shape
            img_and_mask = img_and_mask.reshape((img_and_mask.shape[0:3]))

        if np.random.randint(0,10) == 7:
            img_and_mask = random_rotation(img_and_mask, rg=45, row_axis=0, col_axis=1, channel_axis=2,
                                           fill_mode='constant', cval=0.)

        if np.random.randint(0, 5) == 3:
            img_and_mask = elastic_transform(img_and_mask, alpha=1000, sigma=80, alpha_affine=50)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2,
                                        fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_zoom(img_and_mask, zoom_range=(0.75, 0.75), row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = flip_axis(img_and_mask, axis=1)

        if np.random.randint(0, 10) == 7:
            img_and_mask = flip_axis(img_and_mask, axis=0)

        if np.random.randint(0, 10) == 7:
            salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

        if batch_of_images.ndim == 4:
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[2]//2]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[2]//2:]
        if batch_of_images.ndim == 5:
            img_and_mask = img_and_mask.reshape(orig_shape)
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[2]//2, :]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[2]//2:, :]

        # Ensure the masks did not get any non-binary values.
        batch_of_masks[batch_of_masks > 0.5] = 1
        batch_of_masks[batch_of_masks <= 0.5] = 0

    return(batch_of_images, batch_of_masks)
