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
import logging
from os.path import join, basename
from os import makedirs

import numpy as np
from numpy.random import rand, shuffle
from PIL import Image

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.ioff()

from utils.custom_data_aug import augmentImages, process_image, image_resize2square, image2float_array
from utils.threadsafe import threadsafe_generator

debug = 0
RESOLUTION = 512
COCO_BACKGROUND = (68, 1, 84, 255)
MASK_BACKGROUND = (0,0,0,0)      
GRAYSCALE = True

def change_background_color(img, original_color, new_color):
    '''
    Convert mask color of 4 channels png image to new color 
    '''
    
    r1, g1, b1, a1 = original_color[0], original_color[1], original_color[2], original_color[3]  # Original value
    # mask background color (0,0,0,0)
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
        makedirs(numpy_path)
    except:
        pass
    try:
        makedirs(fig_path)
    except:
        pass

    if not overwrite:
        try:
            with np.load(join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            pass

    try:       
        img = np.array(Image.open(join(img_path, img_name)))
        img = img[:,:,:3]

        if GRAYSCALE == True:
            # Add 5 for each pixel and change resolution on the image.
            img = process_image(img, shift = 5, normalized = False, resolution = RESOLUTION)
                        
            # Translate the image to 24bits grayscale by PILLOW package
            img = image2float_array(img, 16777216-1)  #2^24=16777216

            # Reshape numpy from 2 to 3 dimensions
            img = img.reshape([img.shape[0], img.shape[1], 1])
        else: # Color image with 3 channels
            # Add 5 for each pixel and change resolution on the image.
            img = process_image(img, shift = 5, normalized = True, resolution = RESOLUTION)
            # Keep RGB channel, remove alpha channel
            img = img[:,:,:3]            
            
        if not no_masks:
            # Replace SimpleITK to PILLOW for 2D image support on Raspberry Pi
            mask = np.array(Image.open(join(mask_path, img_name))) # (x,y,4)
            
            mask = image_resize2square(mask, RESOLUTION)
             
            mask = change_background_color(mask, COCO_BACKGROUND, MASK_BACKGROUND) 
            if GRAYSCALE == True:      
                # Only need one channel for black and white      
                mask = mask[:,:,:1]
            else:
                mask = mask[:,:,:1] # keep 3 channels for RGB. Remove alpha channel.



            mask[mask >= 1] = 1 # The mask. ie. class of Person
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


def get_slice(image_data):
    return image_data[2]

@threadsafe_generator
def generate_train_batches(root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):
    # Create placeholders for training
    # (img_shape[1], img_shape[2], args.slices)
    logging.info('\n2d_generate_train_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), (net_input_shape[0], net_input_shape[1], 1)))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                # Read image file from pre-processing image numpy format compression files.
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
                logging.info('\npath_to_np=%s'%(path_to_np))
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                train_img, train_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    logging.info('\nFinished making npz file.')

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
                    logging.error('Error this function currently only supports 2D and 3D data.')
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
                        # [(1, 512, 512, 3), (1, 512, 512, 1)], [(1, 512, 512, 1), (1, 512, 512, 3)]
                        # or [(1, 512, 512, 3), (1, 512, 512, 3)], [(1, 512, 512, 3), (1, 512, 512, 3)]
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
    logging.info('2d_generate_val_batches')
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
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                val_img, val_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(val_img,np.zeros(1)):
                    continue
                else:
                    logging.info('\nFinished making npz file.')
            
            # New added for debugging
            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1: # Only one slices. code can be removed.
                np.random.seed(None)
                subSampAmt = int(rand(1)*(val_img.shape[2]*0.05))
            
            # We don't need indicies in 2D image.        
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
                    logging.error('Error this function currently only supports 2D and 3D data.')
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
    logging.info('load_2D_data.generate_test_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    count = 0
    logging.info('load_2D_data.generate_test_batches: test_list=%s'%(test_list))
    for i, scan_name in enumerate(test_list):
        try:
            scan_name = scan_name[0]
            path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
            with np.load(path_to_np) as data:
                test_img = data['img']
        except:
            logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
            test_img = convert_data_to_numpy(root_path, scan_name, no_masks=True)
            if np.array_equal(test_img,np.zeros(1)):
                continue
            else:
                logging.info('\nFinished making npz file.')

        indicies = np.arange(0, test_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
        for j in indicies:
            if img_batch.ndim == 4:
                img_batch[count, :, :, :] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, :, :, :, 0] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            else:
                logging.error('Error this function currently only supports 2D and 3D data.')
                exit(0)

            count += 1
            if count % batchSize == 0:
                count = 0
                yield (img_batch)

    if count != 0:
        yield (img_batch[:count,:,:,:])
 
@threadsafe_generator
def generate_test_image(test_img, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    '''
    test_img: numpy.array of image data
    
    '''
    # Create placeholders for testing
    logging.info('load_2D_data.generate_test_image')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    count = 0

    #######
    if GRAYSCALE == True:
        test_img = test_img[:,:,:3]
                
        # Add 5 for each pixel and change resolution on the image.
        test_img = process_image(test_img, shift = 5, normalized = False, resolution = RESOLUTION)
                    
        # Translate the image to 24bits grayscale by PILLOW package
        test_img = image2float_array(test_img, 16777216-1)  #2^24=16777216

        # Reshape numpy from 2 to 3 dimensions
        test_img = test_img.reshape([test_img.shape[0], test_img.shape[1], 1])
    else: # Color image with 3 channels
        # Add 5 for each pixel and change resolution on the image.
        test_img = process_image(test_img, shift = 5, normalized = True, resolution = RESOLUTION)
        # Keep RGB channel, remove alpha channel
        test_img = np.reshape(test_img, (1, test_img.shape[0], test_img.shape[1], 4))
        test_img = test_img[:,:,:,:3]
        
    indicies = np.arange(0, test_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
    for j in indicies:
        if img_batch.ndim == 4:
            img_batch[count, :, :, :] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
        elif img_batch.ndim == 5:
            # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
            img_batch[count, :, :, :, 0] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
        else:
            logging.error('Error this function currently only supports 2D and 3D data.')
            exit(0)

        count += 1
        if count % batchSize == 0:
            count = 0
            yield (img_batch)

    if count != 0:
        yield (img_batch[:count,:,:,:])
       