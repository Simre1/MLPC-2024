# src/predict.py

import sys
sys.path.append('../../libs')  # Update this path according to the location of your 'dataset' module
sys.path.append('../../Files')  # Update this path according to the location of your 'dataset' module
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import classes
from models.classifier import AudioClassifierCNN


def get_file_paths(directory, extension=".npy"):
    """
    Get all file paths with a specific extension in a directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]


def sliding_window_prediction(model, input_data, window_size, stride, threshold):
    """
    Perform sliding window prediction using a given model.
    """
    num_windows = ((input_data.shape[2] - window_size) // stride) + 1
    all_predictions = []
    threshold_predictions = []

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window_data = input_data[:, :, start:end]

        y_logit = model(window_data)
        y_pred = torch.softmax(y_logit, dim=1)

        all_predictions.append(y_pred.cpu())

        if torch.max(y_pred) >= threshold:
            threshold_predictions.append((start, end, torch.argmax(y_pred)))

    return all_predictions, threshold_predictions


def display_predictions_heatmap_with_timestamps(all_predictions, classes_dict, window_size, stride, num_displayed_timestamps=10):
    """
    Display a heatmap of predicted probabilities with timestamps.
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


def display_binary_predictions_heatmap_with_timestamps(threshold_predictions, classes_dict, window_size, stride, num_displayed_timestamps=10):
    """
    Display a binary heatmap of predictions exceeding the threshold with timestamps.
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
    data_directory = '../../Files/scence_analysis'

    # Define the model file path
    model_file_path = r"saved_models\model_20240608_141753\model_lr_0.01.pth"

    # Get file paths
    file_paths = get_file_paths(data_directory)
    print("Collected new file paths:", file_paths)

    # Load and preprocess the dataset
    dataset_audio = np.load(file_paths[1])[:, 12:76, :]
    print(f"Dataset shape: {dataset_audio.shape}")
    dataset_audio_tensor = torch.tensor(dataset_audio, dtype=torch.float32)

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
    plot_predictions_heatmap(threshold_predictions, dataset_audio_tensor, classes.REVERSE_CLASSES)


if __name__ == "__main__":
    main()
