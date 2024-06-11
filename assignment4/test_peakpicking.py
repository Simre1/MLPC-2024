import sys
sys.path.append('../../libs')  # Update this path according to the location of your 'dataset' module
sys.path.append('../../Files')  # Update this path according to the location of your 'dataset' module

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


from utils.plot_utils import display_predictions_heatmap_with_timestamps, \
    display_binary_predictions_heatmap_with_timestamps, plot_predictions_heatmap

import utils.classes as classes
from models.classifier import AudioClassifierCNN
from utils.peakpicking_utils import pick_peaks  # Import the peak picking functions
from utils.data_utils import load_scenes_melspect


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


def plot_probabilities_over_time(class_probabilities, class_names, file_path):
    """
    Plot the class probabilities over time.
    """
    plt.figure(figsize=(15, 7))
    for class_idx, probabilities in enumerate(class_probabilities):
        timestamps = [prob[0] for prob in probabilities]
        values = [prob[1] for prob in probabilities]
        plt.plot(timestamps, values, label=class_names[class_idx])
    plt.xlabel('Time Step')
    plt.ylabel('Probability')
    plt.title(f'Class Probabilities Over Time for {os.path.basename(file_path)}')
    plt.legend(loc='upper right')
    plt.show()


def main():
    """
    Main function to perform audio classification using a CNN model and display the results.
    """

    # Define the model file path
    model_file_path = r"saved_models/model_20240608_141753/model_lr_0.01.pth"

    # Load the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Parameters
    sample_rate = 16000  # Example sample rate
    window_size = 44
    stride = 11

    # Load data
    scenes = load_scenes_melspect(max_files=1)

    # Loop through all files and process them
    for scene_name, dataset_audio in scenes:
        print(f"Processing file {scene_name}")
        dataset_audio_tensor = torch.tensor(np.array([dataset_audio]), dtype=torch.float32)

        # Perform sliding window prediction
        all_predictions, threshold_predictions = sliding_window_prediction(
            model, dataset_audio_tensor, window_size=window_size, stride=stride, threshold=0.30  # Adjusted threshold
        )

        # Print some predictions for verification
        print("Sample Predictions:")
        for i in range(10):  # Print the first 10 predictions
            print(all_predictions[i])

        # Process predictions to get detected peaks
        detected_peaks = pick_peaks(all_predictions, sample_rate, stride, height=0.3, distance=5)  # Adjusted parameters

        # Convert class indices to names
        detected_peaks_named = {classes.REVERSE_CLASSES[class_idx]: peaks for class_idx, peaks in detected_peaks.items()}

        # Display the detected peaks for the current file
        print(f"Detected peaks for file {scene_name}:")
        for class_name, timestamps in detected_peaks_named.items():
            print(f"  {class_name}: {timestamps}")

        # Prepare data for the heatmap functions
        timestamps = [pred[0] for pred in all_predictions]
        prob_arrays = [pred[1] for pred in all_predictions]
        all_predictions_array = np.array(prob_arrays)

        # Plot probabilities over time
        class_probabilities = [[] for _ in range(len(classes.CLASSES))]
        for timestamp, probs in all_predictions:
            for class_idx in range(len(probs[0])):
                class_probabilities[class_idx].append((timestamp, probs[0][class_idx]))

        plot_probabilities_over_time(class_probabilities, list(classes.CLASSES.keys()), scene_name)

        # Display predictions for the current file
        #display_predictions_heatmap_with_timestamps(all_predictions_array, list(classes.CLASSES.keys()), window_size=window_size, stride=stride)
        #display_binary_predictions_heatmap_with_timestamps(threshold_predictions, list(classes.CLASSES.keys()), window_size=window_size, stride=stride)
        #plot_predictions_heatmap(threshold_predictions, dataset_audio_tensor, classes.REVERSE_CLASSES)


if __name__ == "__main__":
    main()
