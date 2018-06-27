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

echo "Install Open CV"

sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk2.0-dev libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran

wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.1.zip

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.1.zip

unzip opencv.zip

unzip opencv_contrib.zip

cd ./opencv-3.4.1/
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/SegCaps/opencv_contrib-3.4.1/modules \
    -D BUILD_EXAMPLES=ON ..

make -j4

sudo make install
sudo ldconfig
sudo apt-get update

python -c "import cv2 as cv2; print(cv2.__version__)"

sudo apt-get clean



