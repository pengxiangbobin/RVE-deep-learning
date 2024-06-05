# Introduction
This project is to use U-Net architecture to predict von Mise stress of fiber reinforced composite RVE geometries. 
Using the U-Net structure makes it easy to predict its properties, providing you with insights that can help improve material properties and strength.
# Getting started
This project has been tested on Keras 2.15.0, Python 3.11.5, and Windows 10.
# How U-Net works
1. The first step is to create image and mask for the training and crop them to the specified image size.
2. The second step is to use data augmentation to expand training set.
3. The final step is to train the U-Net model using paired images.

