import numpy as np
from scipy.signal import find_peaks
import utils.classes as classes
import torch

def smooth_probabilities(probabilities, window_size=5):
    smoothed_probs = []
    for i in range(len(probabilities)):
        start = max(0, i - window_size // 2)
        end = min(len(probabilities), i + window_size // 2 + 1)
        window = [prob[1] for prob in probabilities[start:end]]
        smoothed_probs.append(np.mean(window, axis=0))
    return smoothed_probs


def adaptive_peak_picking(smoothed_probs, height=None, distance=1):
    peaks, _ = find_peaks(smoothed_probs, height=height, distance=distance)
    return peaks


def convert_to_seconds(timestamps):
    """
    Convert time steps to seconds based on sample rate and stride.
    """
    return [timestamp * 0.025 for timestamp in timestamps]


def pick_peaks(all_predictions, thresholds, distances):
    # Extract timestamps and probabilities for each class
    num_classes = len(classes.CLASSES)
    class_probabilities = [[] for _ in range(num_classes)]

    for timestamp, probs in all_predictions:
        for class_idx in range(len(probs[0])):
            class_probabilities[class_idx].append((timestamp, probs[0][class_idx]))

    # Apply smoothing and adaptive peak picking for each class
    detected_peaks = {}
    for class_idx, probabilities in enumerate(class_probabilities):
        if classes.class_to_label(class_idx) == "uninteresting":
            continue
        smoothed_probs = smooth_probabilities(probabilities)
        peaks = adaptive_peak_picking(smoothed_probs, height=thresholds[classes.class_to_label(class_idx)], distance=distances[classes.class_to_label(class_idx)])
        peak_timestamps = [probabilities[i][0] for i in peaks]
        peak_timestamps_in_seconds = convert_to_seconds(peak_timestamps)
        detected_peaks[class_idx] = peak_timestamps_in_seconds

    return detected_peaks

def sliding_window_prediction(model, input_data, window_size, stride):
    """
    Perform sliding window prediction using a given model.
    """
    num_windows = ((input_data.shape[2] - window_size) // stride) + 1
    all_predictions = []

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window_data = input_data[:, :, start:end]

        y_logit = model(window_data)
        y_pred = torch.softmax(y_logit, dim=1)

        timestamp = (start + (end - start) / 2)  # Midpoint of the window as the timestamp
        all_predictions.append((timestamp, y_pred.cpu().detach().numpy()))

    return all_predictions

