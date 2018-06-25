#!/bin/bash

echo "Install Tensorflow 1.8 on Python 3.5"
sudo apt-get update
sudo pip3 install --upgrade pip
sudo pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.8.0/tensorflow-1.8.0-cp35-none-linux_armv7l.whl
sudo pip3 uninstall mock
sudo pip3 install mock

echo "Install Keras on Python 3.5"
sudo apt-get install python3-numpy
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install python3-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install gfortran
sudo apt-get install python3-setuptools
sudo apt-get install python3-scipy
sudo apt-get install python3-h5py
sudo pip3 install keras

echo "Install Rest of Packages"
sudo apt-get install python3-matplotlib python3-sklearn python3-pil python3-skimage
reboot
sudo apt-get install python3-cv2
sudo pip3 install jupyter

sudo apt-get clean



