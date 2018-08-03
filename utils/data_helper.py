'''
This program is a helper to separate the codes of 2D and 3D image processing.

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:    Licensed under the Apache License v2.0. http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com

Tasks:
    The program is a helper to separate the codes of 2D and 3D image processing.

    This is a helper file for choosing which dataset functions to create.
'''
import logging
import utils.load_3D_data as ld3D
import utils.load_2D_data as ld2D
from enum import Enum, unique

@unique
class Dataset(Enum):
    luna16 = 1
    mscoco17 = 2
          
def get_generator(dataset):
    if dataset == 'luna16':
        generate_train_batches = ld3D.generate_train_batches
        generate_val_batches = ld3D.generate_val_batches
        generate_test_batches = ld3D.generate_test_batches
    elif dataset == 'mscoco17':
        generate_train_batches = ld2D.generate_train_batches
        generate_val_batches = ld2D.generate_val_batches
        generate_test_batches = ld2D.generate_test_batches
    else:
        logging.error('Not valid dataset!')
        return None, None, None
    return generate_train_batches, generate_val_batches, generate_test_batches
    
