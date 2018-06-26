'''


This is a helper file for choosing which dataset functions to create.
'''
import logging
from load_data import *
import load_3D_data as ld3D
import load_2D_data as ld2D
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
    
# def get_test_batches_generator(dataset):
#     generate_test_batches = None
#     if dataset == 'luna16':
#         generate_test_batches = ld3D.generate_test_batches
#     elif dataset =='mscoco17':
#         generate_test_batches = ld2D.generate_test_batches
#     else:
#         logging.error('Not valid dataset!')
#         return None    
#     return generate_test_batches
# 
# def get_train_batches_generator(dataset):
#     generate_train_batches = None
#     if dataset == 'luna16':
#         generate_train_batches = ld3D.generate_train_batches
#     elif dataset =='mscoco17':
#         generate_train_batches = ld2D.generate_train_batches
#     else:
#         logging.error('Not valid dataset!')
#         return None    
#     return generate_train_batches 
#    
#  
# def get_val_batches_generator(dataset):
#     generate_val_batches = None
#     if dataset == 'luna16':
#         generate_val_batches = ld3D.generate_val_batches
#     elif dataset =='mscoco17':
#         generate_val_batches = ld2D.generate_val_batches
#     else:
#         logging.error('Not valid dataset!')
#         return None    
#     return generate_val_batches  
