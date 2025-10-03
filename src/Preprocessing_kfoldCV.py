import os
from glob import glob
import torch
#import numpy as np
#import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    Orientationd,
    # Data Augmentation transforms
    RandRotated,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    OneOf             # OneOf is a MONAI transform that  provides the ability to randomly choose one transform out of a list of callables with pre-defined probabilities for each.
)
from sklearn.model_selection import KFold
from monai.data import Dataset, DataLoader, CacheDataset, NibabelReader
from monai.utils.misc import first, set_determinism

def define_files(in_path):
    '''
    this function returns the dictionaries of training and testing
    will be used in the preprocessing
    '''
# Glob will create a list of paths for each file
    train_imgs = sorted(glob(os.path.join(in_path, 'imagesTr', '*.nii.gz')))
    train_lbls = sorted(glob(os.path.join(in_path, 'labelsTr', '*.nii.gz')))
    test_imgs =  sorted(glob(os.path.join(in_path, 'imagesTs', '*.nii.gz')))
    
    print(f"Found {len(train_imgs)} training images")
    print(f"Found {len(train_lbls)} training labels")
    print(f"Found {len(test_imgs)} testing images")
# Create dictionary 
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_imgs, train_lbls)]
    test_files = [{"image": img} for img in test_imgs]
    
    return train_files, test_files
    
def get_transforms(train_files, val_files, test_files, pixdim=(1.5, 1.5, 4.0), a_min=0.0, a_max=1448.00, spatial_size=[128, 128,16], cache=True, augment=True, batch_size=1):

    """
    This function is for preprocessing, it contains basic transforms and data augmentation transforms.
    It's very important that only the training set gets augmented, validation and obviously testing only get basic transforms
    16 slices is the minimum slices found in the training volumes. 
    Note: UNet is composed of both a encoder and a decoder, if you put a number of slices that's not a power of 2 
    you will get errors in the rounding of the tensors dimensions
    pixdim: pixel dimension for resample 
    a_min and a_max: Intensity scale
    spatial_size: target spatial dimension, 14 slices because it's the minimum viable number of slices found in the dataset
    cache(bool): wether to use Cachedataset
    augment(bool): wether to use data augmentation transforms
    batch_size: Batch size for data loaders

    Monai documentation: https://monai.io/docs.html

    will return loaders for training
    """

    set_determinism(seed = 0)

    #basic preprocessing transforms
    basic_T = [
    LoadImaged(keys = ['image', 'label'], reader=NibabelReader()), 
    # Basic transforms
    EnsureChannelFirstd(keys = ['image', 'label']),
    Spacingd(keys = ['image', 'label'], pixdim = pixdim, mode = ('bilinear','nearest') ),
    Orientationd(keys = ['image', 'label'], axcodes = 'RAS'),
    ScaleIntensityRanged(keys = 'image', a_min = a_min , a_max = a_max, b_min=0.0, b_max=1.0, clip=True), # scaling the intensity apply it only on the image
    CropForegroundd(keys = ['image','label'], source_key = 'image'), # need to specifiy which one is the image
    Resized(keys = ['image','label'], spatial_size = spatial_size),
    ]
    # data augmenting transoforms
    augment_T = []
    if augment:
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


    # Preprocessing transformations
    #Training
    train_transforms = Compose(
    basic_T +
    augment_T +
    [ToTensord(keys = ['image', 'label'])]
    )
    # Validation
    val_transforms = Compose(
    basic_T +
    [ToTensord(keys = ['image', 'label'])]
    )

    # Testing transformation use EXCLUSIVELY the basic transform, we have only images 
    test_transforms = Compose(
    [
        LoadImaged(keys = 'image', reader=NibabelReader()), 

        EnsureChannelFirstd(keys = 'image'),
        Spacingd(keys = 'image', pixdim = pixdim, mode = 'bilinear' ),
        Orientationd(keys = 'image', axcodes = 'RAS'),
        ScaleIntensityRanged(keys = 'image', a_min = a_min , a_max = a_max, b_min=0.0, b_max=1.0, clip=True), # scaling the intensity apply it only on the image
        CropForegroundd(keys = 'image', source_key = 'image'), # need to specifiy which one is the image
        Resized(keys = 'image', spatial_size = spatial_size),
        
        ToTensord(keys = 'image')
    ]
    )

    # create datasets and data loaders
    # CacheDataset: performance optimization feature of MONAI that stores transformed data in memory
    # no need to apply transform for every epoch
    # note: eats a lot of RAM for bigger dataset may cause out-of-memory errors (es. set cache_rate = 0.5) 
    if cache:
        train_ds = CacheDataset(data = train_files, transform = train_transforms, cache_rate = 1.0)
        val_ds = CacheDataset(data = val_files, transform = val_transforms, cache_rate = 1.0) 
        test_ds = CacheDataset(data = test_files, transform = test_transforms, cache_rate = 1.0)

        train_loader = DataLoader(train_ds, batch_size = 1)
        val_loader = DataLoader(val_ds, batch_size = 1)
        test_loader = DataLoader(test_ds, batch_size = 1)
    else:
        train_ds = Dataset(data = train_files, transform = train_transforms)
        val_ds = Dataset(data = val_files, transform = val_transforms, cache_rate = 1.0)
        test_ds = Dataset(data = test_files, transform = test_transforms)

        train_loader = DataLoader(train_ds, batch_size = 1)
        val_loader = DataLoader(val_ds, batch_size = 1)
        test_loader = DataLoader(test_ds, batch_size = 1)

    return train_loader, val_loader, test_loader

def create_kfold_split(train_files, n_splits=5, shuffle = True, random_state=42):
    '''
    create Kfold splits of the training set with scikit - learn
    shuffle: decide to shuffle or not before split,
    random_state: random seed
    '''
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kfold.split(train_files))
    
    
