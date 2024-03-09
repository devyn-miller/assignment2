# load_preprocess.py

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import warnings

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
TRAINING_PATH = './assignment2data/train/'
TESTING_PATH = './assignment2data/validation/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

def load_data():
    train_ids = next(os.walk(TRAINING_PATH))[1]
    test_ids = next(os.walk(TESTING_PATH))[1]
    return train_ids, test_ids

def preprocess_data():
    # Check if the data has already been preprocessed and saved
    if os.path.exists('x_train.npy') and os.path.exists('y_train.npy'):
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
    else:
        train_ids, test_ids = load_data()
        print('Getting and resizing training images ... ')
        x_train = np.zeros((len(train_ids), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
        y_train = np.zeros((len(train_ids), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool_)
                
        # Re-sizing our training images to 128 x 128
        # Note sys.stdout prints info that can be cleared unlike print.
        # Using TQDM allows us to create progress bars
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = TRAINING_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:IMAGE_CHANNELS]
            img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
            x_train[n] = img
            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.bool_)
            
            # Now we take all masks associated with that image and combine them into one single mask
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', 
                                            preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            # y_train is now our single mask associated with our image
            y_train[n] = mask

        # Save the resized images and masks to disk
        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)

    # Repeat the process for test images if necessary
    if os.path.exists('x_test.npy') and os.path.exists('sizes_test.npy'):
        x_test = np.load('x_test.npy')
        sizes_test = np.load('sizes_test.npy')
    else:
        test_ids = next(os.walk(TESTING_PATH))[1]
        print('Getting and resizing test images ... ')
        x_test = np.zeros((len(test_ids), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
        sizes_test = []
        sys.stdout.flush()

        # Here we resize our test images
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = TESTING_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:IMAGE_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='constant', preserve_range=True)
            x_test[n] = img

        # Save the resized test images and their sizes to disk
        np.save('x_test.npy', x_test)
        np.save('sizes_test.npy', sizes_test)

    print('Done!')
    return x_train, y_train, x_test, sizes_test

def visualize_sample_images(X, Y, n_samples=5):
    """Visualizes n_samples of images with their corresponding masks."""
    x_train, y_train, x_test, sizes_test = preprocess_data()
    # Illustrate the train images and masks
    plt.figure(figsize=(20,16))
    x, y = 12,4
    for i in range(y):  
        for j in range(x):
            plt.subplot(y*2, x, i*2*x+j+1)
            pos = i*120 + j*10
            plt.imshow(x_train[pos])
            plt.title('Image #{}'.format(pos))
            plt.axis('off')
            plt.subplot(y*2, x, (i*2+1)*x+j+1)
            
            #We display the associated mask we just generated above with the training image
            plt.imshow(np.squeeze(y_train[pos]))
            plt.title('Mask #{}'.format(pos))
            plt.axis('off')
            
    plt.show()

def display_images_and_masks_from_arrays(images, masks, num_imgs=5):
    fig, axs = plt.subplots(num_imgs, 2, figsize=(10, num_imgs * 5))
    for i in range(num_imgs):
        img = images[i]
        mask = masks[i]
        
        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(mask.squeeze(), cmap='gray')
        axs[i, 1].set_title('Mask')
        
    for ax in axs.flat:
        ax.label_outer()
        
    plt.show()

def plot_image_size_distribution(images, title):
    widths, heights = [], []
    for image in images:
        heights.append(image.shape[0])
        widths.append(image.shape[1])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='skyblue')
    plt.title('Width Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='lightgreen')
    plt.title('Height Distribution')
    
    plt.suptitle(title)
    plt.show()

