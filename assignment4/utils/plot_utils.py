# src/utils/plot_utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, prediction_file):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix for {}".format(prediction_file))

    base_filename = os.path.basename(prediction_file)
    filename = os.path.splitext(base_filename)[0]
    save_path = os.path.join(os.path.dirname(prediction_file), f"{filename}_confusion_matrix.jpg")

    plt.savefig(save_path)
    # plt.show()


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
