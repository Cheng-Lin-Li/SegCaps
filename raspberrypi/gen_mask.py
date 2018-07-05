#!/usr/bin/env python
# encoding: utf-8
'''
Generate image mask by trained model.
Tasks: Input an image file and output a mask image file.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:    Licensed under the Apache License v2.0. http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com
@version:    1.0

@create:    June 27, 2018
@updated:   June 27, 2018 

Tasks:
The program implementation will classify input image by a trained model and generate mask image as 
image segmentation results.


Data:

Currently focus on person category data.

'''
from threading import Thread
import os, argparse, logging
from os import path
from os.path import join
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
import matplotlib.pyplot as plt
import sys
from Cython.Compiler.PyrexTypes import cap_length
# Add the ptdraft folder path to the sys.path list
sys.path.append('../')
from utils.model_helper import create_model

# from data_helper import *
from utils.load_2D_data import generate_test_image
from utils.custom_data_aug import image_resize2square
from datetime import datetime
import cv2

FILE_MIDDLE_NAME = 'train'
IMAGE_FOLDER = 'imgs'
MASK_FOLDER = 'masks'
RESOLUTION = 512 # Resolution of the input for the model.



class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
        
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

        
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color=(0, 1, 0), alpha=0.5):
    """apply mask to image"""
    
    #convert to floating point
    image = image.astype(np.float)
    image /= 255.0

    #convert to floating point
    mask = mask.astype(np.float) #mask=[512, 512]
    mask /= 255.0
    #set transparency as alpha
    mask *= alpha
    zero_3d_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=mask.dtype)

    field3d_mask = zero_3d_mask + mask[..., np.newaxis]    

    #make a green overlay
    mask_color = np.ones(image.shape, dtype = np.float) * color
    
    #green over original image
    out = mask_color * field3d_mask + image * (1.0 - field3d_mask)

    return out


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
#     n_instances = boxes.shape[0]
#     colors = random_colors(n_instances)
    colors = random_colors(1)
#     if not n_instances:
#         print('NO INSTANCES TO DISPLAY')
#     else:
#         assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

#     for i, color in enumerate(colors):
#         if not np.any(boxes[i]):
#             continue
# 
#         y1, x1, y2, x2 = boxes[i]

    mask = masks

    image = apply_mask(image, mask, colors[0])
#     image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#     image = cv2.putText(
#             image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
#         )

    return image


def threshold_mask(raw_output, threshold): #raw_output 3d:(119, 512, 512)
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    logging.info('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0

    #all_labels 3d:(119, 512, 512)
    all_labels = measure.label(raw_output)
    # props 3d: region of props=>list(_RegionProperties:<skimage.measure._regionprops._RegionProperties object>) 
    # with bbox. 
    props = measure.regionprops(all_labels) 
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        # if the largest is way larger than the second largest
        if props[0].area / props[1].area > 5:  
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1
    # threshold_mask: 3d=(119, 512, 512)
    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)
        
    return thresholded_mask, props[0]

class SegCaps():
    def __init__(self, args, net_input_shape):
        '''
        Create evaluation model and load the pre-train weights for inference.
        '''
        self.net_input_shape = net_input_shape
        weights_path = join(args.weights_path)
        # Create model object in inference mode.
        _, eval_model, _ = create_model(args, net_input_shape) 
    
        # Load weights trained on MS-COCO
        eval_model.load_weights(weights_path)
        self.model = eval_model
               
        
    def detect(self, img_list, verbose = False):
        
        result = []
        r = dict()
        
        print(str(datetime.now()))
        for img_data in img_list:
            output_array = self.model.predict_generator(generate_test_image(img_data,
                                                                          self.net_input_shape,
                                                                          batchSize=1,
                                                                          numSlices=1,
                                                                          subSampAmt=0,
                                                                          stride=1),
                                                    steps=1, max_queue_size=1, workers=1,
                                                    use_multiprocessing=False, verbose=1)
            output = output_array[0][:,:,:,0]
            threshold_level = 0
            output_bin, props = threshold_mask(output, threshold_level)  
                      
        print(str(datetime.now()))    
        r['masks'] = output_bin[0,:,:]
        r['class_ids'] = 0
        r['rois'] = props
        r['scores'] = 0.75
        result.append(r)
        return result 
        
def change_res(cap, width, height):
    '''
    Change resolutions of OpenCV video capture stream.
    '''
    cap.set(3, width)
    cap.set(4, height)
    
    return cap
    

if __name__ == '__main__':
    '''
    Main program for images segmentation by mask image.
    Example command:
    $python3 gen_mask --input_file ../data/image/train1.png --net segcapsr3 --model_weight ../data/saved_models/segcapsr3/dice16-255.hdf5
    '''
    
    parser = argparse.ArgumentParser(description = 'Mask image by segmentation algorithm')
#     parser.add_argument('--data_root_dir', type = str, required = True,
#                         help = 'The root directory for your data.')
#     parser.add_argument('--input_file', type = str, required = True,
#                         help='The input image filename under your --data_root_dir/imgs/ folder.')
    parser.add_argument('--net', type = str.lower, default = 'segcapsr3',
                        choices = ['segcapsr3', 'segcapsr1', 'capsbasic', 'unet', 'tiramisu'],
                        help = 'Choose your network.')    
    parser.add_argument('--weights_path', type = str, required = True,
                        help = '/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--num_class', type = int, default = 2, 
                        help = 'Number of classes to segment. Default is 2. If number of classes > 2, '
                            ' the loss function will be softmax entropy and only apply on SegCapsR3'
                            '** Current version only support binary classification tasks.')    
    parser.add_argument('--which_gpus', type = str, default = '0',
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type = int, default = -1,
                        help = 'Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')    


    args = parser.parse_args()
    class_names = ['person']
    net_input_shape = (RESOLUTION, RESOLUTION, 1)
    model = SegCaps(args, net_input_shape)
    
    # grab a pointer to the video stream and initialize the FPS counter
    print('[INFO] sampling frames from webcam...')

    stream = cv2.VideoCapture(0)
    stream.set(3,512)
    stream.set(4,512)
    # these 3 lines can control fps, frame width and height.   
#     stream.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION)
#     stream.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION)    
    stream.set(cv2.CAP_PROP_FPS, 0.025)     
    fps = FPS().start()
    
    # loop over some frames
    while fps._numFrames < 10000:
        # grab the frame from the stream and resize it to have a maximum
        # width of 400 pixels
        (grabbed, frame) = stream.read()
        frame = image_resize2square(frame, RESOLUTION) # frame = (512, 512, 3)
 
        # check to see if the frame should be displayed to our screen
        results = model.detect([frame], verbose=0)
        r = results[0] #r['masks'] = [512, 512]
        frame = apply_mask(frame, r['masks'], (1,0,0), 0.7)
#         frame = display_instances(
#             frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
#         )
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        else:
            pass
        # update the FPS counter
        fps.update()
 
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    stream.release()
    cv2.destroyAllWindows()  
    