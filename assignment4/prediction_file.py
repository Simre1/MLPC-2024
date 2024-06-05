import sys

sys.path.append('../libs')  # Update this path according to the location of your 'dataset' module
sys.path.append('../Files')

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import classes
from classifier_architectures import AudioClassifierCNN


def get_file_paths(directory, extension=".npy"):
    """
    Get all file paths with a specific extension in a directory.
    
    Parameters:
    - directory: Path to the directory containing files.
    - extension: File extension to filter by (default is ".npy").
    
    Returns:
    - List of file paths with the specified extension.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]


def sliding_window_prediction(model, input_data, window_size, stride, threshold):
    """
    Perform sliding window prediction using a given model.

    Parameters:
    - model: The CNN classifier model.
    - input_data: The longer frame input data (shape: 1*64*854).
    - window_size: The size of the sliding window (number of timesteps).
    - stride: The stride of the sliding window.
    - threshold: The threshold for predicted probabilities.

    Returns:
    - all_predictions: A list of predicted probabilities for all windows and classes.
    - threshold_predictions: A list of tuples (start, end, prediction_tensor) for windows where the maximum predicted probability exceeds the threshold.
    """

    # Calculate the number of windows
    num_windows = ((input_data.shape[2] - window_size) // stride) + 1

    # Initialize empty lists to store predictions
    all_predictions = []
    threshold_predictions = []

    # Slide the window over the input data and make predictions
    for i in range(num_windows):
        # Extract the current window
        start = i * stride
        end = start + window_size
        window_data = input_data[:, :, start:end]

        # Make prediction using the model
        y_logit = model(window_data)

        # Turn predictions from logits -> prediction probabilities -> predictions labels
        y_pred = torch.softmax(y_logit, dim=1)

        # Store the predicted probability tensor in all_predictions
        all_predictions.append(y_pred.cpu())

        # Check if the maximum predicted probability is above the threshold
        if torch.max(y_pred) >= threshold:
            # Store the predicted probability tensor along with start and end indices
            threshold_predictions.append((start, end, torch.argmax(y_pred)))

    return all_predictions, threshold_predictions


def display_predictions(all_predictions):
    """
    Display predictions for each window.
    
    Parameters:
    - all_predictions: List of predicted probabilities for all windows and classes.
    """
    for i, pred_tensor in enumerate(all_predictions):
        pred_array = pred_tensor.detach().numpy()
        print(f"Window {i + 1}:")
        for class_index, class_pred in enumerate(pred_array[0]):
            print(f"Class {class_index}: {class_pred:.4f}")
        print()


def display_predictions_heatmap_with_timestamps(all_predictions, classes_dict, window_size, stride,
                                                num_displayed_timestamps=10):
    """
    Display a heatmap of predicted probabilities with timestamps.
    
    Parameters:
    - all_predictions: List of predicted probabilities for all windows and classes.
    - classes_dict: Dictionary mapping class indices to class labels.
    - window_size: The size of the sliding window (number of timesteps).
    - stride: The stride of the sliding window.
    - num_displayed_timestamps: Number of timestamps to display on the x-axis.
    """
    all_predictions_array = np.array([pred.detach().numpy() for pred in all_predictions])
    num_windows = len(all_predictions)
    timestamps = [(i * stride, i * stride + window_size) for i in range(num_windows)]
    step = max(num_windows // num_displayed_timestamps, 1)

    plt.figure(figsize=(10, 6))
    plt.imshow(all_predictions_array.squeeze().T, cmap='Greys', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Probability')
    plt.xlabel('Time (frames)')
    plt.ylabel('Class Index')
    plt.title('Predicted Probabilities Heatmap')

    plt.xticks(ticks=np.arange(0, num_windows, step), labels=np.arange(0, num_windows * stride, step * stride))

    if classes_dict:
        plt.yticks(ticks=np.arange(len(classes_dict)), labels=list(classes_dict.keys()))

    plt.show()


def display_binary_predictions_heatmap_with_timestamps(threshold_predictions, classes_dict, window_size, stride,
                                                       num_displayed_timestamps=10):
    """
    Display a binary heatmap of predictions exceeding the threshold with timestamps.
    
    Parameters:
    - threshold_predictions: List of tuples (start, end, prediction_tensor) for windows where the maximum predicted probability exceeds the threshold.
    - classes_dict: Dictionary mapping class indices to class labels.
    - window_size: The size of the sliding window (number of timesteps).
    - stride: The stride of the sliding window.
    - num_displayed_timestamps: Number of timestamps to display on the x-axis.
    """
    num_timestamps = ((threshold_predictions[-1][1] - threshold_predictions[0][0]) // stride) + 1
    heatmap = np.zeros((len(classes_dict), num_timestamps))

    for prediction in threshold_predictions:
        start, end, label = prediction
        timestamp_start = (start - threshold_predictions[0][0]) // stride
        timestamp_end = (end - threshold_predictions[0][0]) // stride
        heatmap[label, timestamp_start:timestamp_end] = 1

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, cmap='binary', aspect='auto')
    plt.xlabel('Time (frames)')
    plt.ylabel('Class Index')
    plt.title('Binary Heatmap, Threshold prob.: 0.8')

    step = max(num_timestamps // num_displayed_timestamps, 1)
    plt.xticks(ticks=np.arange(0, num_timestamps, step),
               labels=np.arange(threshold_predictions[0][0], threshold_predictions[-1][1] + 1, step * stride))

    if classes_dict:
        plt.yticks(ticks=np.arange(len(classes_dict)), labels=list(classes_dict.keys()))

    plt.show()


def plot_predictions_heatmap(predictions, input_data, reverse_class_labels):
    """
    Plot the predictions on the spectrogram of the input data.
    
    Parameters:
    - predictions: List of tuples (start, end, prediction_tensor) for windows where the maximum predicted probability exceeds the threshold.
    - input_data: The original input data.
    - reverse_class_labels: Dictionary mapping class indices to class labels.
    """
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(input_data.squeeze(0), aspect='auto', origin='lower', cmap='viridis')
    for prediction in predictions:
        start, end, label = prediction
        if label != 0:
            plt.axvline(x=start, color='r', linestyle='--')
            plt.axvline(x=end, color='r', linestyle='--')
            mid_point = (start + end) / 2
            class_key = reverse_class_labels[label.item()]
            plt.text(mid_point, input_data.shape[1] + 1, label.item(), color='red', ha='center', va='bottom')

    plt.xlabel('Timesteps')
    plt.ylabel('Features')
    plt.colorbar(label='Amplitude')
    plt.show()


def main():
    """
    Main function to perform audio classification using a CNN model and display the results.
    """
    # Directory containing the new .npy files
    data_directory = '../Files/scence_analysis'

    # Get file paths
    file_paths = get_file_paths(data_directory)
    print("Collected new file paths:", file_paths)

    # Load and preprocess the dataset
    dataset_audio = np.load(file_paths[1])[:, 12:76, :]
    print(f"Dataset shape: {dataset_audio.shape}")
    dataset_audio_tensor = torch.tensor(dataset_audio, dtype=torch.float32)

    # Define the model file path
    model_file_path = r"saved_models\model_lr_0.001.pth"

    # Load the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Perform sliding window prediction
    all_predictions, threshold_predictions = sliding_window_prediction(
        model, dataset_audio_tensor, window_size=44, stride=11, threshold=0.80
    )

    # Display predictions
    display_predictions_heatmap_with_timestamps(all_predictions, classes.CLASSES, window_size=44, stride=11)
    display_binary_predictions_heatmap_with_timestamps(threshold_predictions, classes.CLASSES, window_size=44,
                                                       stride=11)
    plot_predictions_heatmap(threshold_predictions, dataset_audio_tensor, classes.REVERSE_CLASSES)


if __name__ == "__main__":
    main()
