'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which dataset functions to create.
'''
import load_3D_data as ld3D # load_data, split_data
import load_2D_data as ld2D

def get_data_helper(dataset):
    data_helper = None
    if dataset == 'luna16':
        data_helper = ld3D.image_3D()
    elif dataset =='mscoco17':
        data_helper = ld2D.image_2D()
    else:
        print('Not valid dataset!')
        return None    
    return data_helper