import sys
sys.path.append('../../libs')  # Update this path according to the location of your 'dataset' module
sys.path.append('../../Files')  # Update this path according to the location of your 'dataset' module

import os
import torch
import numpy as np

from utils.plot_utils import display_predictions_heatmap_with_timestamps, \
    display_binary_predictions_heatmap_with_timestamps, plot_predictions_heatmap

import classes
from models.classifier import AudioClassifierCNN
from peakpicking_utils import process_predictions  # Import the peak picking functions

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

        timestamp = (start + (end - start) / 2)  # Midpoint of the window as the timestamp
        all_predictions.append((timestamp, y_pred.cpu().detach().numpy()))

        if torch.max(y_pred) >= threshold:
            threshold_predictions.append((start, end, torch.argmax(y_pred)))

    return all_predictions, threshold_predictions


def main():
    """
    Main function to perform audio classification using a CNN model and display the results.
    """
    # Directory containing the new .npy files
    data_directory = r"C:\Users\naglm\OneDrive\Uni\AI\UE Machine Learning and Pattern Classification\MLPC-2024\Files\scene_analysis"

    # Define the model file path
    model_file_path = r"saved_models\model_20240608_141753\model_lr_0.001.pth"

    # Get file paths
    file_paths = get_file_paths(data_directory)
    print("Collected new file paths:", file_paths)

    # Load the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Parameters
    sample_rate = 16000  # Example sample rate
    window_size = 44
    stride = 11

    # Loop through all files and process them
    for file_path in file_paths:
        # Load and preprocess the dataset
        dataset_audio = np.load(file_path)[:, 12:76, :]
        print(f"Processing file {file_path} with dataset shape: {dataset_audio.shape}")
        dataset_audio_tensor = torch.tensor(dataset_audio, dtype=torch.float32)

        # Perform sliding window prediction
        all_predictions, threshold_predictions = sliding_window_prediction(
            model, dataset_audio_tensor, window_size=window_size, stride=stride, threshold=0.60
        )

        # Process predictions to get detected peaks
        detected_peaks = process_predictions(all_predictions, sample_rate, stride)

        # Convert class indices to names
        detected_peaks_named = {classes.REVERSE_CLASSES[class_idx]: peaks for class_idx, peaks in detected_peaks.items()}

        # Display the detected peaks for the current file
        print(f"Detected peaks for file {file_path}:")
        for class_name, timestamps in detected_peaks_named.items():
            print(f"  {class_name}: {timestamps}")

        # Prepare data for the heatmap functions
        timestamps = [pred[0] for pred in all_predictions]
        prob_arrays = [pred[1] for pred in all_predictions]
        all_predictions_array = np.array(prob_arrays)

        # Display predictions for the current file
        #display_predictions_heatmap_with_timestamps(all_predictions_array, list(classes.CLASSES.keys()), window_size=window_size, stride=stride)
        #display_binary_predictions_heatmap_with_timestamps(threshold_predictions, list(classes.CLASSES.keys()), window_size=window_size, stride=stride)
        #plot_predictions_heatmap(threshold_predictions, dataset_audio_tensor, classes.REVERSE_CLASSES)


if __name__ == "__main__":
    main()
