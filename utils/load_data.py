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

import logging

from os.path import join, basename
from os import makedirs
from glob import glob
import csv
from sklearn.model_selection import KFold
import numpy as np

import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm #Progress bar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()

debug = 0



def load_data(root, split):
    # Load the training and testing lists
    with open(join(root, 'split_lists', 'train_split_' + str(split) + '.csv'), 'r') as f:
        reader = csv.reader(f)
        training_list = list(reader)

    with open(join(root, 'split_lists', 'test_split_' + str(split) + '.csv'), 'r') as f:
        reader = csv.reader(f)
        testing_list = list(reader)

    new_training_list, validation_list = train_test_split(training_list, test_size = 0.1, random_state = 7)
    if new_training_list == []: # if training_list only have 1 image file.
        new_training_list = validation_list
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
        logging.warning('Class weight file {} not found.\nComputing class weights now. This may take '
              'some time.'.format(class_weight_filename))
        train_data_list, _, _ = load_data(root, str(split))
        value = compute_class_weights(root, train_data_list)
        np.save(class_weight_filename,value)
        logging.warning('Finished computing class weights. This value has been saved for this training split.')
        return value


def split_data(root_path, num_splits):
    mask_list = []
    for ext in ('*.mhd', '*.hdr', '*.nii', '*.png'): #add png file support
        mask_list.extend(sorted(glob(join(root_path,'masks',ext)))) # check imgs instead of masks

    assert len(mask_list) != 0, 'Unable to find any files in {}'.format(join(root_path,'masks'))

    outdir = join(root_path,'split_lists')
    try:
        makedirs(outdir)
    except:
        pass

    if num_splits == 1:
        # Testing model, training set = testing set = 1 image
        train_index = test_index = mask_list
        with open(join(outdir,'train_split_' + str(0) + '.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            print('basename=%s'%([basename(mask_list[0])]))
            writer.writerow([basename(mask_list[0])])
        with open(join(outdir,'test_split_' + str(0) + '.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([basename(mask_list[0])])
     
    else:
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



