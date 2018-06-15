#!/usr/bin/env python
# encoding: utf-8
'''
Data crawler for MS COCO 2017 semantic segmentation.
Tasks: Download specific category images from MSCOCO web and generate pixel level masking image files on PNG format.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:    Licensed under the Apache License v2.0. http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com
@version:    1.0

@create:    June 13, 2018
@updated:   June 13, 2018 

Tasks:
The program implementation leverage pycocotools to batch download images by specific category and generate mask files for image segmentation tasks.


Data:

Currently focus on person category data.

'''

import argparse

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab, os
from tqdm import tqdm

FILE_MIDDLE_NAME = 'train'
IMAGE_FOLDER = 'imgs'
MASK_FOLDER = 'masks'

def main(args):
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    plt.ioff()
    
    data_dir = args.data_root_dir
    category_list = list(args.category)
    annFile = args.annotation_file   
    num = args.number
    file_name = ''
 
    #Create path for output
    output_image_path = data_dir+'/'+IMAGE_FOLDER
    if not os.path.isdir(output_image_path):
        os.makedirs (output_image_path) 
    output_mask_path = data_dir+'/'+MASK_FOLDER   
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
        plt.axis('off')
        file_name = output_image_path+'/'+FILE_MIDDLE_NAME+str(i)+'.png'
        plt.imsave(file_name, I)
         
        # Get annotation
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
         
        # Generate mask
        for j in range(len(anns)):
             mask += coco.annToMask(anns[j])
         
        # Background color = (R,G,B)=(68, 1, 84)
        # save the mask image
        file_name = output_mask_path+'/m_'+FILE_MIDDLE_NAME+str(i)+'.png'
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
    