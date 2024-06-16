# src/predict.py

import sys
sys.path.append('../../Files')  # Update this path according to the location of your 'dataset' module
import os
import torch
import numpy as np

from utils.plot_utils import display_predictions_heatmap_with_timestamps, \
    display_binary_predictions_heatmap_with_timestamps, plot_predictions_heatmap

import utils.classes as classes

from utils.data_utils import load_scenes_melspect

from models.classifier import AudioClassifierCNN

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


def main():
    """
    Main function to perform audio classification using a CNN model and display the results.
    """
    # Define the model file path
    model_file_path = r"saved_models/model_20240615_181836/model_lr_0.001.pth"

    # Load the scene data
    dataset_audio = load_scenes_melspect(max_files=1)

    print(f"Type of dataset_audio: {type(dataset_audio)}")
    print(f"First element type: {type(dataset_audio[0])}")
    print(f"First element: {dataset_audio[0]}")

    # Extract the audio data (the second element in the tuple)
    audio_data = dataset_audio[0][1]

    print(f"Dataset shape: {audio_data.shape}")

    # Add a batch dimension to the audio data
    audio_data = np.expand_dims(audio_data, axis=0)

    print(f"Dataset shape with batch dimension: {audio_data.shape}")

    dataset_audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # Load the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Perform sliding window prediction
    all_predictions, threshold_predictions = sliding_window_prediction(
        model, dataset_audio_tensor, window_size=44, stride=11, threshold=0.50
    )

    # Display predictions
    display_predictions_heatmap_with_timestamps(all_predictions, classes.CLASSES, window_size=44, stride=11)
    display_binary_predictions_heatmap_with_timestamps(threshold_predictions, classes.CLASSES, window_size=44,
                                                       stride=11)
    plot_predictions_heatmap(threshold_predictions, dataset_audio_tensor, classes.REVERSE_CLASSES)


if __name__ == "__main__":
    main()


