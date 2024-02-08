#%%
from PIL import UnidentifiedImageError
# %% Package import
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import shutil


# Define directory paths
dataset_dir = './yolov7/dataset'
train_dir = './yolov7/dataset/train'
test_dir = './yolov7/dataset/test'

# Create directories if they don't exist
for dir_path in [dataset_dir, train_dir, test_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

os.listdir('./yolov7/dataset')

#%% List of classes
classes = ['axial_t1wce_2_class', 'coronal_t1wce_2_class', 'sagittal_t1wce_2_class']

# %% Copy data for each class
for class_name in classes:
    dataset_path = f'./brain_tumor_detection/{class_name}'
    train_path = f'./yolov7/dataset/train/{class_name}'
    test_path = f'./yolov7/dataset/test/{class_name}'

    # Create directories if they don't exist
    for dir_path in [os.path.join(train_path, 'images'),
                     os.path.join(train_path, 'labels'),
                     os.path.join(test_path, 'images'),
                     os.path.join(test_path, 'labels')]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    ! cp {dataset_path}/images/train/* {train_path}/images/
    ! cp {dataset_path}/labels/train/* {train_path}/labels/
    ! cp {dataset_path}/images/test/* {test_path}/images/
    ! cp {dataset_path}/labels/test/* {test_path}/labels/

os.listdir(f'./yolov7/dataset/train/{class_name}/images')

#%% List files in directories
train_images = os.listdir(train_path+'/images')
test_images = os.listdir(test_path+'/images')

plt.figure(figsize=(20, 10))

for i, c in enumerate(np.random.randint(0, len(train_images), size=10), start=1):
    plt.subplot(2, 5, i)
    try:
        im = plt.imread(train_path+'/images/'+train_images[c])
        plt.imshow(im, cmap='gray')
    except (OSError, UnidentifiedImageError) as e:
        print(f"Error loading image {train_images[c]}: {e}")
    plt.axis('off')


# %%
os.listdir('./yolov7/data')
# %%