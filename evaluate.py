# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from architecture_train import my_iou_metric, iou_metric
from load_preprocess import preprocess_data
import matplotlib.cm as cm
import random
from skimage.io import imshow


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3




def predict():
    model = load_model('/home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_2.h5', 
                    custom_objects={'my_iou_metric': my_iou_metric})
    x_train, y_train = preprocess_data()
    # the first 90% was used for training
    preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)

    # the last 10% used as validation
    preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)

    #preds_test = model.predict(x_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    return preds_train_t, preds_val_t

def plot_masks():
    x_train, y_train = preprocess_data()
    preds_train_t, preds_val_t = predict()
    # Ploting our predicted masks
    ix = random.randint(0, 602)
    plt.figure(figsize=(20,20))

    # Our original training image
    plt.subplot(131)
    imshow(x_train[ix])
    plt.title("Image")

    # Our original combined mask  
    plt.subplot(132)
    imshow(np.squeeze(y_train[ix]))
    plt.title("Mask")

    # The mask our U-Net model predicts
    plt.subplot(133)
    imshow(np.squeeze(preds_train_t[ix] > 0.5))
    plt.title("Predictions")
    plt.show()

    # Ploting our predicted masks
    ix = random.randint(602, 668)
    plt.figure(figsize=(20,20))

    # Our original training image
    plt.subplot(121)
    imshow(x_train[ix])
    plt.title("Image")

    # The mask our U-Net model predicts
    plt.subplot(122)
    ix = ix - 603
    imshow(np.squeeze(preds_val_t[ix] > 0.5))
    plt.title("Predictions")
    plt.show()
    return ix

def classification_report():
    x_train, y_train = preprocess_data()
    preds_train_t, preds_val_t = predict()
    ix = plot_masks()
    iou_metric(np.squeeze(y_train[ix]), np.squeeze(preds_train_t[ix]), print_table=True)
    iou_metric(np.squeeze(y_train[ix]), np.squeeze(preds_val_t[ix]), print_table=True)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    plt.imshow(superimposed_img)
    plt.show()

def three_best_three_worst():
    x_train, y_train = preprocess_data()
    model = load_model('model-unet.h5', custom_objects={'my_iou_metric': my_iou_metric})
    ious = []
    for img, mask in zip(x_train, y_train):
        pred_mask = model.predict(img[np.newaxis, ...])[0]  # Assuming 'model' is your trained model
        iou = iou_metric(mask, pred_mask)
        ious.append(iou)
    ious = np.array(ious)


def display_images_with_iou(indices, ious):
    x_train, y_train = preprocess_data()
    model = load_model('model-unet.h5', custom_objects={'my_iou_metric': my_iou_metric})
    fig, axs = plt.subplots(1, len(indices), figsize=(20, 5))
    for i, idx in enumerate(indices):
        img = x_train[idx]
        mask = y_train[idx]
        pred_mask = model.predict(img[np.newaxis, ...])[0]
        
        axs[i].imshow(np.squeeze(img), 'gray', interpolation='none')
        axs[i].imshow(np.squeeze(pred_mask) > 0.5, 'jet', alpha=0.5, interpolation='none')
        axs[i].set_title(f'IoU: {ious[idx]:.4f}')
        axs[i].axis('off')
    plt.show()
    best_three_indices = np.argsort(ious)[-3:]
    worst_three_indices = np.argsort(ious)[:3]

    # Display the best and worst performing images
    return display_images_with_iou(best_three_indices, ious), display_images_with_iou(worst_three_indices, ious)