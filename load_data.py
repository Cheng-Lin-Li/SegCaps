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

MASK_FILE_PRE = 'm_'

import threading
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

class image():
    def __init__(self, dataset='mscoco17'):
        self.dataset = dataset
        

    def image_resize2square(self, image, desired_size = None):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        old_size = image.shape[:2]
    
        # if both the width and height are None, then return the
        # original image
        if desired_size is None:
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
    
    
    def load_data(self, root, split):
        # Load the training and testing lists
        with open(join(root, 'split_lists', 'train_split_' + str(split) + '.csv'), 'r') as f:
            reader = csv.reader(f)
            training_list = list(reader)
    
        with open(join(root, 'split_lists', 'test_split_' + str(split) + '.csv'), 'r') as f:
            reader = csv.reader(f)
            testing_list = list(reader)
    
        new_training_list, validation_list = train_test_split(training_list, test_size=0.1, random_state=7)
    
        return new_training_list, validation_list, testing_list
    
    def compute_class_weights(self, root, train_data_list):
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
            img = sitk.GetArrayFromImage(sitk.ReadImage(join(root, 'masks', MASK_FILE_PRE+img_name[0])))
            for slic in img:
                if not np.any(slic):
                    continue
                else:
                    p = np.count_nonzero(slic)
                    pos += p
                    neg += (slic.size - p)
    
        return neg/pos
    
    def load_class_weights(self, root, split):
        class_weight_filename = join(root, 'split_lists', 'train_split_' + str(split) + '_class_weights.npy')
        try:
            return np.load(class_weight_filename)
        except:
            print('Class weight file {} not found.\nComputing class weights now. This may take '
                  'some time.'.format(class_weight_filename))
            train_data_list, _, _ = self.load_data(root, str(split))
            value = self.compute_class_weights(root, train_data_list)
            np.save(class_weight_filename,value)
            print('Finished computing class weights. This value has been saved for this training split.')
            return value
    
    
    def split_data(self, root_path, num_splits=4):
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
            with open(join(outdir,'train_split_' + str(n) + '.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in train_index:
                    print('basename=%s'%([basename(mask_list[i])]))
                    writer.writerow([basename(mask_list[i])])
            with open(join(outdir,'test_split_' + str(n) + '.csv'), 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in test_index:
                    writer.writerow([basename(mask_list[i])])
            n += 1
    
    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    
    def augmentImages(self, batch_of_images, batch_of_masks):
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
                img_and_mask = self.flip_axis(img_and_mask, axis=1)
    
            if np.random.randint(0, 10) == 7:
                img_and_mask = self.flip_axis(img_and_mask, axis=0)
    
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


    @threadsafe_generator
    def generate_train_batches(self, root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                               stride=1, downSampAmt=1, shuff=1, aug_data=1):
        # Create placeholders for training
        # (img_shape[1], img_shape[2], args.slices)
        print('==>base_generate_train_batches')        
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
                    print('=>path_to_np=%s'%(path_to_np))
                    with np.load(path_to_np) as data:
                        train_img = data['img']
                        train_mask = data['mask']
                except:
                    print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                    train_img, train_mask = self.convert_data_to_numpy(root_path, scan_name)
                    if np.array_equal(train_img,np.zeros(1)):
                        continue
                    else:
                        print('\nFinished making npz file.')
    
                if numSlices == 1:
                    subSampAmt = 0
                elif subSampAmt == -1 and numSlices > 1:
                    np.random.seed(None)
                    subSampAmt = int(rand(1)*(train_img.shape[2]*0.05))
    
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
                            img_batch, mask_batch = self.augmentImages(img_batch, mask_batch)
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
                    img_batch[:count,...], mask_batch[:count,...] = self.augmentImages(img_batch[:count,...],
                                                                                  mask_batch[:count,...])
                if net.find('caps') != -1:
                    yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                           [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
                else:
                    yield (img_batch[:count,...], mask_batch[:count,...])
    
    @threadsafe_generator
    def generate_val_batches(self, root_path, val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                             stride=1, downSampAmt=1, shuff=1):
        print('==>base_generate_val_batches')
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
                    val_img, val_mask = self.convert_data_to_numpy(root_path, scan_name)
                    if np.array_equal(val_img,np.zeros(1)):
                        continue
                    else:
                        print('\nFinished making npz file.')
    
                if numSlices == 1:
                    subSampAmt = 0
                elif subSampAmt == -1 and numSlices > 1:
                    np.random.seed(None)
                    subSampAmt = int(rand(1)*(val_img.shape[2]*0.05))
    
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
    def generate_test_batches(self, root_path, test_list, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                              stride=1, downSampAmt=1):
        # Create placeholders for testing
        print('load_3D_data.generate_test_batches')
        img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
        count = 0
        print('load_3D_data.generate_test_batches: test_list=%s'%(test_list))
        for i, scan_name in enumerate(test_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
                with np.load(path_to_np) as data:
                    test_img = data['img']
            except:
                print('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                test_img = self.convert_data_to_numpy(root_path, scan_name, no_masks=True)
                if np.array_equal(test_img,np.zeros(1)):
                    continue
                else:
                    print('\nFinished making npz file.')
    
            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(rand(1)*(test_img.shape[2]*0.05))
    
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
        