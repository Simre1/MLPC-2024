import numpy as np
from scipy.signal import find_peaks


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


def convert_to_seconds(timestamps, sample_rate, stride):
    """
    Convert time steps to seconds based on sample rate and stride.
    """
    return [timestamp * stride / sample_rate for timestamp in timestamps]


def process_predictions(all_predictions, sample_rate, stride, height=0.7, distance=5):
    # Extract timestamps and probabilities for each class
    num_classes = len(all_predictions[0][1][0])
    class_probabilities = [[] for _ in range(num_classes)]

    for timestamp, probs in all_predictions:
        for class_idx in range(len(probs[0])):
            class_probabilities[class_idx].append((timestamp, probs[0][class_idx]))

    # Apply smoothing and adaptive peak picking for each class
    detected_peaks = {}
    for class_idx, probabilities in enumerate(class_probabilities):
        smoothed_probs = smooth_probabilities(probabilities)
        peaks = adaptive_peak_picking(smoothed_probs, height=height, distance=distance)
        peak_timestamps = [probabilities[i][0] for i in peaks]
        peak_timestamps_in_seconds = convert_to_seconds(peak_timestamps, sample_rate, stride)
        detected_peaks[class_idx] = peak_timestamps_in_seconds

    return detected_peaks
