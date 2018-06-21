#!/usr/bin/env python
# encoding: utf-8
'''
Data crawler for MS COCO 2017 semantic segmentation.
Tasks: Download specific category images from MSCOCO web and generate pixel level masking image files on PNG format.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:    Licensed under the Apache License v2.0. http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com
@version:    1.1

@create:    June 13, 2018
@updated:   June 20, 2018 

Tasks:
The program implementation leverage pycocotools to batch download images by specific category and generate mask files for image segmentation tasks.


Data:

Currently focus on person category data.

'''

import argparse
from os.path import join, basename
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab, os
from tqdm import tqdm
import cv2

FILE_MIDDLE_NAME = 'train'
IMAGE_FOLDER = 'imgs'
MASK_FOLDER = 'masks'
RESOLUTION = 512 # Resolution of the input for the model.

def image_resize2square(image, desired_size = None):
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

def main(args):
#     pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    plt.ioff()
    
    data_dir = args.data_root_dir
    category_list = list(args.category)
    annFile = args.annotation_file   
    num = args.number
    file_name = ''
 
    #Create path for output
    output_image_path = join(data_dir, IMAGE_FOLDER)
    if not os.path.isdir(output_image_path):
        os.makedirs (output_image_path) 
    output_mask_path = join(data_dir, MASK_FOLDER) 
    if not os.path.isdir(output_mask_path):
        os.makedirs (output_mask_path)    
 
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
     
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=category_list);
 
    # Get image id list from categories.
    imgIds = coco.getImgIds(catIds=catIds );
    print('Image Generating...')
    for i in tqdm(range(num)):
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
     
        # use url to load image
        I = io.imread(img['coco_url'])
        I = image_resize2square(I, RESOLUTION)
        plt.axis('off')
        file_name = join(output_image_path, FILE_MIDDLE_NAME+str(i)+'.png')
        plt.imsave(file_name, I)
         
        # Get annotation
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
         
        # Generate mask
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])
         
        # Background color = (R,G,B)=[68, 1, 84]
        # save the mask image
        mask = image_resize2square(mask, RESOLUTION)
        file_name = join(output_mask_path, FILE_MIDDLE_NAME+str(i)+'.png')
        plt.imsave(file_name, mask)
         
    print('Image Generation Complete !')
    return True

if __name__ == '__main__':
    '''
    Main program for MS COCO 2017 annotation mask images generation.
    Example command:
    $python getcoco17 --data_root_dir ./data --category person dog --annotation_dir './annotations/instances_val2017.json --number 10'
    '''
    
    parser = argparse.ArgumentParser(description='Download COCO 2017 image Data')
    parser.add_argument('--data_root_dir', type=str, required=False,
                        help='The root directory for your data.')
    parser.add_argument('--category', nargs='+', type=str, default='person',
                        help='MS COCO object categories list (--category person dog cat). default value is person')
    parser.add_argument('--annotation_file', type=str, default='./instances_val2017.json',
                        help='The annotation json file directory of MS COCO object categories list. file name should be instances_val2017.json')    
    parser.add_argument('--number', type=int, default=10,
                        help='The total number of images you want to download.')    

    arguments = parser.parse_args()
    
    main(arguments)
    