# Capsules for Object Segmentation (SegCaps)
### by [Rodney LaLonde](https://rodneylalonde.wixsite.com/personal) and [Ulas Bagci](http://www.cs.ucf.edu/~bagci/)

### Modified by [Cheng-Lin Li](https://cheng-lin-li.github.io/about/)
### Objectives: Build up a pipeline for Object Segmentation experiments on SegCaps with not only 3D CT images (LUNA 16) but also 2D color images (MS COCO 2017).

## This repo is the clone the implementation of SegCaps from official site with restructure and enhancements.

The original paper for SegCaps can be found at https://arxiv.org/abs/1804.04241.

The original source code can be found at https://github.com/lalonderodney/SegCaps

Author's project page for this work can be found at https://rodneylalonde.wixsite.com/personal/research-blog/capsules-for-object-segmentation.


## Getting Started Guide

The program was modified to support python 3.6 on Ubuntu 18.04 and Windows 10.

### 1. Install Required Packages on Ubuntu / Windows
This repo of code is written for Keras using the TensorFlow backend. 
You may need to adjust requirements.txt file according to your environment (CPU only or GPU for tensorflow installation). 

Please install all required packages before using programs.

```bash
pip install -r requirements.txt
```
### You may need to install additional library in Ubuntu version 17 or above version.
Following steps will resolve below issue.
```text
ImportError: libjasper.so.1: cannot open shared object file: No such file or directory
```
```bash
sudo apt-get update
sudo apt-get install libjasper-dev
```

### 2. Download this repo to your own folder
Example: repo folder name ~/SegCaps/

### 3. Make your data directory.
Below commands:

  3-1. Create root folder name 'data' in the repo folder. All models, results, etc. are saved to this root directory.

  3-2. Create imgs and masks folders for image and mask files. 

```bash
mkdir data
chmod 755 data
cd ./data
mkdir imgs
mkdir masks
chmod 755 *
cd ..
```

### 4. Select Your dataset

#### 4-1. Test the result on original LUNA 16 dataset.
  1. Go to [LUng Nodule Analysis 2016 Grand-Challenges website](https://luna16.grand-challenge.org/)
  2. Get an account by registration.
  3. Join the 'LUNA 16' challenge by click 'All Challenges' on the tab of top. Click the 'Join' and goto 'Download' section to get your data.
  4. copy your image files into BOTH ./data/imgs and ./data/masks folders.

#### 4-2. Test on Microsoftsoft Common Objects in COntext (MS COCO) dataset 2017.
The repo include a crawler program to download your own class of images for training.
But you have to download the annotation file first.
[Microsoft COCO 2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
There are two JSON files contain in the zip file. Extract them into a folder. 

In this example the folder is under ~/SegCaps/annotations/

Example 1: Download 10 images and mask files with 'person' class from MS COCO dataset.

```bash
cd ./cococrawler
$python3 getcoco17.py --data_root_dir ../data --category person --annotation_file ./annotations/instances_val2017.json --number 10
```

Example 2: Download image IDs 22228, and 178040 with mask images for only person class.
```bash
cd ./cococrawler
$python3 getcoco17.py --data_root_dir ../data/coco --category person --annotation_file ./annotations/instances_train2017.json  --number 10 --id 22228 178040
```

You can choose multiple classes if you want. Just specify category of each class by space. 

Example: --category person dog cat


### 5. Train your model
#### 5-1 Main File

From the main file (main.py) you can train, test, and manipulate the segmentation capsules of various networks. Simply set the ```--train```, ```--test```, or ```--manip flags``` to 0 or 1 to turn these off or on respectively. The argument ```--data_root_dir``` is the only required argument and should be set to the directory containing your *imgs* and *masks* folders. There are many more arguments that can be set and these are all explained in the main.py file. 

#### Example command: Train SegCaps R3 on MS COCO dataset without GPU support.

```bash
python3 main.py --train=1 --test=0 --manip=0 --initial_lr 0.1 --net segcapsr3 --loss dice --data_root_dir=data --which_gpus=-2 --gpus=0 --dataset mscoco17 
```
### 6. Program Descriptions
  1. main.py: The entry point of this project.
  2. train.py: The major training module.
  3. test.py: The major testing module.
  4. manip.py: The manipulate module of the model.

### 7. Program Structures:
```text
----SegCaps  (Project folder)
    |
    \-cococrawler (Crawler program folder)
    |   \-annotations (Folder of Microsoft COCO annotation files)
    \-data  (The root folder of program output)
    |   \-imgs (Folder of training and testing images)
    |   \-masks (Folder of training and testing masking data)
    |   \-np_files (Folder to store processed image and mask files in numpy form.)
    |   \-split_lists (Folder for training and testing image splits list)
    |   \-logs (Training logs)
    |   \-plots (Trend diagram for Training period. Only generate after the training completed )
    |   \-figs (Conver image to numpy format, part of images stored for checking)
    |   \-saved_models (All model weights will be stored under this folder)
    |   \-results (Test result images will be stored in this folder)
    |
    \-models (Reference model files: Unet and DenseNet)
    |
    \-segcapsnet (main modules for Capsule nets and SegCaps)
    |
    \-utils (image loader, loss functions, metrics, image augmentation, and thread safe models)
    |
    \-notebook (Some experiment notebooks for reference)
    |
    \-raspberrypi (A video streaming capture program integrated with SegCaps for segmentation task) 
    |
    \-installation (Installation shell for Raspberry Pi)
    |
    \-imgs (image file for this readme)
```

### 8. Install package on Raspberry Pi 3
### The section is under constructing. The SegCaps model cannot fit into the memory of Raspberry Pi 3 so far.
#### Download tensorflow pre-compile version for ARM v7.
Tensorflow for ARM - Github Repo:
https://github.com/lhelontra/tensorflow-on-arm/releases

installation instruction.

https://medium.com/@abhizcc/installing-latest-tensor-flow-and-keras-on-raspberry-pi-aac7dbf95f2

#### OpenCV installation on Raspberry Pi 3
https://www.alatortsev.com/2018/04/27/installing-opencv-on-raspberry-pi-3-b/

## 9. TODO List:
  1. Execute programs on LUNA 16 dataset. Done. Jun 11

    1-1. Porting program from python 2.7 to 3.6 (Jun 11)

    1-2. Execute manipulation function. (Jun 11)

    1-2. Execute test function on one image without pre-trained weight(Jun 11)

    1-3. Execute train function on 3 images. (Jun 12)

    1-4. Execute test function on trained model (Jun 12)

    1-5. Display original image and result mask image. (Jun 12)

    1-6. Identify input image mask format. (Jun 14)

  2. Find right dataset for person/cat/dog segmentation. Candidate dataset is MS COCO. Done. 6/12 COCO 2017

    2-1. Identify COCO stuff 2017 as target dataset. (Jun 15)

    2-2. Download annotation files for COCO 2017. (Jun 15)

  3. Test existing program on color images.

    3-1. Generate single class mask on COCO masked image data.(Jun 13)

    3-2. Convert the image mask format to background=0, objects=1. (Jun 18)

    3-3. Convert the color image to gray scale image (Jun 18)

    3-3. Feed the color images with single classes mask to model for training. (Jun 21)

  4. Pipeline up:

    4-1. Modify code to support experiments.(Jun 25)

      4-1-1. Models persistent by version with configuration and dataset. (Jun 26)

      4-1-2. Notebook folder build up to store experiment results.

    4-2. Test pipeline (Jun 27)

  5. Modify program for color images. (Jun 27)

  6. Model training (Jun 27)
  
  7. Integrate model with webcam. (Jul 3)

### Citation

This project based on the official codebase of Capsules for Object Segmentation:
```
@article{lalonde2018capsules,
  title={Capsules for Object Segmentation},
  author={LaLonde, Rodney and Bagci, Ulas},
  journal={arXiv preprint arXiv:1804.04241},
  year={2018}
}
```

### Questions or Comments
For this modification version, please email me at clark.cl.li@gmail.com

For the original implementation, please direct any questions or comments to the author. You can either comment on the [project page](https://rodneylalonde.wixsite.com/personal/research-blog/capsules-for-object-segmentation), or email author directly at lalonde@knights.ucf.edu.















## Original README.md description:
<img src="imgs/qualitative1.png" width="900px"/>

## Condensed Abstract
Convolutional neural networks (CNNs) have shown remarkable results over the last several years for a wide range of computer vision tasks. A new architecture recently introduced by [Sabour et al., referred to as a capsule networks with dynamic routing](https://arxiv.org/abs/1710.09829), has shown great initial results for digit recognition and small image classification. Our work expands the use of capsule networks to the task of object segmentation for the first time in the literature. We extend the idea of convolutional capsules with *locally-connected routing* and propose the concept of *deconvolutional capsules*. Further, we extend the masked reconstruction to reconstruct the positive input class. The proposed convolutional-deconvolutional capsule network, called **SegCaps**, shows strong results for the task of object segmentation with substantial decrease in parameter space. As an example application, we applied the proposed SegCaps to segment pathological lungs from low dose CT scans and compared its accuracy and efficiency with other U-Net-based architectures. SegCaps is able to handle large image sizes (512 x 512) as opposed to baseline capsules (typically less than 32 x 32). The proposed SegCaps reduced the number of parameters of U-Net architecture by **95.4%** while still providing a better segmentation accuracy.

## Baseline Capsule Network for Object Segmentation

<img src="imgs/baselinecaps.png" width="900px"/>

## SegCaps (R3) Network Overview

<img src="imgs/segcaps.png" width="900px"/>

## Quantative Results on the LUNA16 Dataset

| Method           | Parameters | Split-0 (%) | Split-1 (%) | Split-2 (%) | Split-3 (%) | Average (%) |
|:---------------- |:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| U-Net            | 31.0 M     | 98.353      | 98.432      | 98.476      | **98.510**  | 98.449      |
| Tiramisu         | 2.3 M      | 98.394      | 98.358      | **98.543**  | 98.339      | 98.410      |
| Baseline Caps    | 1.7 M      | 82.287      | 79.939      | 95.121      | 83.608      | 83.424      |
| SegCaps (R1)     | **1.4 M**  | 98.471      | 98.444      | 98.401      | 98.362      | 98.419      |
| **SegCaps (R3)** | **1.4 M**  | **98.499**  | **98.523**  | 98.455      | 98.474      | **98.479**  |

## Results of Manipulating the Segmentation Capsule Vectors

<img src="imgs/manip_cropped.png" width="900px"/>

