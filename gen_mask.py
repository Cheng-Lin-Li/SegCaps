#!/usr/bin/env python
# encoding: utf-8
'''
Generate image mask by trained model.
Tasks: Input an image file and output a mask image file.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:    Licensed under the Apache License v2.0. http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com
@version:    1.2

@create:    June 27, 2018
@updated:   July 06, 2018 

Tasks:
The program implementation will classify input image by a trained model and generate mask image as 
image segmentation results.


Data:

Currently focus on person category data.

Reference:
https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py

'''
from threading import Thread
import argparse
from os.path import join
import numpy as np
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('../')
from utils.model_helper import create_model

# from data_helper import *
from utils.load_2D_data import generate_test_image
from utils.custom_data_aug import image_resize2square
from test import threshold_mask
from datetime import datetime
import cv2

FILE_MIDDLE_NAME = 'train'
IMAGE_FOLDER = 'imgs'
MASK_FOLDER = 'masks'
RESOLUTION = 512 # Resolution of the input for the model.
ARGS = None
NET_INPUT = None


class FPS:
    '''
    Calculate Frame per Second
    '''
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
    '''
    Leverage thread to read video stream to speed up process time.
    '''
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


def apply_mask(image, mask):
    """apply mask to image"""
    
    
    redImg = np.zeros(image.shape, image.dtype)
    redImg[:,:] = (0,0,255)
    redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
    cv2.addWeighted(redMask, 1, image, 1, 0, image)

    return image



class segmentation_model():
    '''
    Model construction class for prediction
    '''
    def __init__(self, args, net_input_shape):
        '''
        Create evaluation model and load the pre-train weights for inference.
        '''
        self.net_input_shape = net_input_shape
        weights_path = join(args.weights_path)
        # Create model object in inference mode but Disable decoder layer.
        _, eval_model, _ = create_model(args, net_input_shape, enable_decoder = False) 
    
        # Load weights trained on MS-COCO by name because part of output layers are disable.
        eval_model.load_weights(weights_path, by_name=True)
        self.model = eval_model
               
        
    def detect(self, img_list, verbose = False):   
        result = []
        r = dict()
        
        for img_data in img_list:
            output_array = self.model.predict_generator(generate_test_image(img_data,
                                                                          self.net_input_shape,
                                                                          batchSize=1,
                                                                          numSlices=1,
                                                                          subSampAmt=0,
                                                                          stride=1),
                                                    steps=1, max_queue_size=1, workers=4,
                                                    use_multiprocessing=False, verbose=1)
            output = output_array[:,:,:,0]
            threshold_level = 0
            output_bin = threshold_mask(output, threshold_level)      
        r['masks'] = output_bin[0,:,:]
        
        # If you want to test the masking without prediction, mark out above line and unmark below line.
        # Below line is make a dummy masking to test the speed.
#         r['masks'] = np.ones((512, 512), np.int8) # Testing 
        result.append(r)
        return result 


if __name__ == '__main__':
    '''
    Main program for images segmentation by mask image.
    Example command:
    $python3 gen_mask --input_file ../data/image/train1.png --net segcapsr3 --model_weight ../data/saved_models/segcapsr3/dice16-255.hdf5
    '''
    
    parser = argparse.ArgumentParser(description = 'Mask image by segmentation algorithm')

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
    net_input_shape = (RESOLUTION, RESOLUTION, 1)
    model = segmentation_model(args, net_input_shape)
    
    
#     # grab a pointer to the video stream and initialize the FPS counter
#     print('[INFO] sampling frames from webcam...')
#     cap = cv2.VideoCapture(0)    
      
    # these 3 lines can control fps, frame width and height.   
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION)    
#     cap.set(cv2.CAP_PROP_FPS, 0.1)     
#     fps = FPS().start()
 
    # created a *threaded* video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from webcam...")
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
     
    # loop over some frames
    while fps._numFrames < 10000:
        # grab the frame from the capture stream and resize it to have a maximum
#         (grabbed, frame) = cap.read()
        frame = vs.read()
        frame = image_resize2square(frame, RESOLUTION) # frame = (512, 512, 3)
  
        # check to see if the frame should be displayed to our screen
        results = model.detect([frame], verbose=0)
        r = results[0] #r['masks'] = [512, 512]
        frame = apply_mask(frame, r['masks'])
         
        cv2.imshow("Frame", frame)
        # Press q or ESC to stop the video
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
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
    vs.release()
    cv2.destroyAllWindows()  
    