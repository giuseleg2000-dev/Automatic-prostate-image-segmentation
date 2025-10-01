# Automatic-prostate-image-segmentation
A deep learning project focused on 3D medical image segmentation using U-Net architectures, with emphasis on handling limited datasets through robust validation strategies and data augmentation.


## 🎯 Project Highlights
Multiple U-Net Architectures: Standard U-Net, Attention U-Net, and U-Net++
Validation: 5-fold cross-validation for reliable performance estimation
Data Augmentation Pipeline: Comprehensive augmentations tailored for medical imaging
Small Dataset Optimization: Techniques to maximize learning from limited data (31 volumes)

## Table of Contents
* Overview
* Dataset Challenge
* Architecture
* Project Structure
* Results
* Key Learnings

## Overview
3D medical image segmentation with MONAI framework and Pytorch. Addressing a common challenge in medical deep learning tasks: achieving reasonable performance with limited training data available

### Key aspects
* **3D Volume Processing**: Full 3D convolutional networks for volumetric segmentation
* **K-Fold Cross Validation**: 5-fold CV to ensure robust model evaluation
* **Flexible Architecture**: Easy switching between different U-Net variants
* **Comprehensive Preprocessing**: MONAI-based transforms including resampling, intensity normalization, and cropping
* **Data Augmentation**: Random rotations, flips, intensity variations, and noise injection

## Dataset
[decathlon open source data set](http://medicaldecathlon.com/)
* Training volumes: 32 3D medical images with labels
* Test volumes: separate test set
* channels: 2
* classes: 2
### Why it matters
Working with only 31 training volumes represents a realistic constraint in medical imaging:

* Medical data acquisition is expensive and time-consuming
* Expert annotation requires significant clinical expertise
* Privacy regulations limit data sharing
* Many clinical datasets fall in this size range

This project aims to demonstrate practical approaches to handling such constraints rather than relying on massive datasets.
## Architecture
### Models implemented
1. standard UNet
2. Attention UNet
3. UNet++
### Loss Function
* Dice Loss: optimized for segmentation overlap

### Data Augmentation strategy
Data augmentation transform are applied only to training dataset with the following transforms
``` python
    augment_T = [
        # Apply random rotation (range_* is in radians) and flip [0,1] = flip along x and y axes
        RandRotated(keys = ['image','label'], range_x=0.3, range_y=0.3, range_z=0.1, prob=1.0, keep_size=True, mode = ('bilinear','nearest')),
        RandFlipd(keys = ['image','label'], spatial_axis = [0,1], prob=0.5),
        # apply one of these transformation to the intensity of the image,
        # factors is in % (15%), offset is ±15%, gamma is contrast adjustment
        OneOf([
            RandScaleIntensityd(keys = 'image', factors=0.15, prob=1.0),
            RandShiftIntensityd(keys = 'image', offsets=0.15, prob=1.0),
            RandAdjustContrastd(keys = 'image', gamma=(0.7,1.3), prob=1.0)
        ], weights = [0.4, 0.4, 0.2]),
        # Noise and smoothing
        RandGaussianNoised(keys = 'image', std = 0.03, prob = 0.15),
    ]
```
## Structure
```
├── train.py                      # Main training script with K-fold CV
├── Preprocessing_kfoldCV.py      # Data loading and augmentation pipeline
├── Util.py                       # Training loop and utility functions
├── MODEL/                        # Saved models and metrics
│   ├── fold1/
│   │   ├── best_metric_model.pth
│   │   ├── loss_train.npy
│   │   ├── loss_val.npy
│   │   ├── metric_train.npy
│   │   └── metric_val.npy
│   ├── fold2/
│   └── ...
├── Img/                          # Dataset directory
│   ├── imagesTr/                 # Training images
│   ├── labelsTr/                 # Training labels
│   └── imagesTs/                 # Test images
└── README.md
```
