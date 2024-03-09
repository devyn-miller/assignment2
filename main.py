from load_preprocess import preprocess_data, load_data, visualize_sample_images, display_images_and_masks_from_arrays, plot_image_size_distribution
from architecture_train import build_unet, train_model, plot_training_history
from evaluate import predict, plot_masks, classification_report, make_gradcam_heatmap, display_gradcam, three_best_three_worst, display_images_with_iou

def main():
    # Load and preprocess data
    x_train, y_train, x_test, sizes_test = preprocess_data()
    train_ids, test_ids = load_data()
    
    # Visualize data
    visualize_sample_images(x_train, y_train)
    display_images_and_masks_from_arrays(x_train[:5], y_train[:5])
    plot_image_size_distribution(x_train, 'Train Image Size Distribution')
    plot_image_size_distribution(x_test, 'Test Image Size Distribution')
    
    # Build and train the U-Net model
    model = build_unet()
    model, history = train_model(x_train, y_train, x_test, sizes_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    predict()
    plot_masks()
    classification_report()
    
    # Grad-CAM and IoU evaluations
    # Assuming you have an img_array for Grad-CAM
    img_array = x_test[0]  # Placeholder, replace with actual data
    heatmap = make_gradcam_heatmap(img_array, model, 'last_conv_layer_name')
    display_gradcam(img_array, heatmap)
    three_best_three_worst()
    
    # Display images with IoU
    # Assuming you have indices and ious calculated
    indices = [0, 1, 2]  # Placeholder, replace with actual data
    ious = [0.9, 0.8, 0.85]  # Placeholder, replace with actual data
    display_images_with_iou(indices, ious)

if __name__ == "__main__":
    main()