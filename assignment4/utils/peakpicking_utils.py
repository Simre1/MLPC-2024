import numpy as np
import scipy.signal as signal
import torch
import matplotlib.pyplot as plt

def normalize_signal(audio_signal):
    """
    Normalize the audio signal to have zero mean and unit variance.

    Args:
        audio_signal (numpy array): The raw audio signal data.

    Returns:
        numpy array: The normalized audio signal.
    """
    return (audio_signal - np.mean(audio_signal)) / np.std(audio_signal)

def compute_spectrogram(audio_signal, fs=16000):
    """
    Compute the spectrogram of the audio signal.

    Args:
        audio_signal (numpy array): The raw or normalized audio signal data.
        fs (int, optional): The sampling frequency of the audio signal (default is 16000).

    Returns:
        tuple:
            numpy array: Array of sample frequencies.
            numpy array: Array of segment times.
            numpy array: Spectrogram of the audio signal.
    """
    frequencies, times, Sxx = signal.spectrogram(audio_signal, fs=fs)
    return frequencies, times, Sxx

def smooth_signal(time_domain_signal, kernel_size=5):
    """
    Apply a median filter to smooth the time-domain signal.

    Args:
        time_domain_signal (numpy array): The time-domain representation of the signal (e.g., the sum of spectrogram frequencies).
        kernel_size (int, optional): The size of the filter kernel (default is 5).

    Returns:
        numpy array: The smoothed time-domain signal.
    """
    return signal.medfilt(time_domain_signal, kernel_size=kernel_size)

def detect_peaks(smoothed_signal, height=0.5, distance=200):
    """
    Detect significant peaks in the smoothed signal.

    Args:
        smoothed_signal (numpy array): The smoothed time-domain signal.
        height (float, optional): Minimum height of peaks (default is 0.5).
        distance (int, optional): Minimum distance between peaks (default is 200).

    Returns:
        numpy array: Indices of detected peaks in the smoothed signal.
    """
    peaks, _ = signal.find_peaks(smoothed_signal, height=height, distance=distance)
    return peaks

def extract_features_around_peaks(audio_signal, peaks, window_size=(175, 44)):
    """
    Extract features around each detected peak.

    Args:
        audio_signal (numpy array): The raw or normalized audio signal.
        peaks (numpy array): Indices of detected peaks.
        window_size (tuple, optional): Size of the window around each peak to extract features (height, width).

    Returns:
        numpy array: Extracted features around each peak.
    """
    features = []
    height, width = window_size
    for peak in peaks:
        start = max(0, peak - width // 2)
        end = min(audio_signal.shape[1], peak + width // 2)
        feature = audio_signal[:, start:end]
        if feature.shape[1] < width:
            feature = np.pad(feature, ((0, 0), (0, width - feature.shape[1])), 'constant')
        if feature.shape == (height, width):  # Ensure the feature has the correct dimensions
            features.append(feature)
    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")
    return features

def classify_peaks(classifier, features):
    """
    Classify the features extracted from each peak using a trained classifier.

    Args:
        classifier (PyTorch model): The trained classification model.
        features (numpy array): Extracted features around each peak.

    Returns:
        torch.tensor: Predicted labels for each peak.
    """
    if features.size == 0:
        print("No features extracted, skipping classification.")
        return torch.tensor([])

    feature_tensors = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    feature_tensors = feature_tensors.unsqueeze(2).unsqueeze(3)  # Ensure it has shape (batch_size, channels, height, width)
    print(f"Feature tensors shape: {feature_tensors.shape}")
    with torch.no_grad():
        predictions = classifier(feature_tensors)
        predicted_labels = torch.argmax(predictions, dim=1)
    return predicted_labels

def map_peaks_to_keywords(peaks, times, smoothed_signal, classifier, keywords, threshold=0.5):
    """
    Map detected peaks to keywords based on a threshold and classifier predictions.

    Args:
        peaks (numpy array): Indices of detected peaks.
        times (numpy array): Array of segment times from the spectrogram.
        smoothed_signal (numpy array): The smoothed time-domain signal.
        classifier (PyTorch model): The trained classification model.
        keywords (list): List of keyword labels.
        threshold (float, optional): Threshold for peak significance (default is 0.5).

    Returns:
        list: List of tuples containing the timestamp and detected keyword.
    """
    detected_keywords = []
    features = extract_features_around_peaks(smoothed_signal, peaks)
    predicted_labels = classify_peaks(classifier, features)

    for peak, label in zip(peaks, predicted_labels):
        if smoothed_signal[peak] > threshold:
            time = times[peak]
            keyword = keywords[label.item()]
            detected_keywords.append((time, keyword))

    return detected_keywords

def plot_peaks(times, smoothed_signal, peaks, detected_keywords):
    """
    Plot the smoothed signal and detected peaks, annotating detected keywords.

    Args:
        times (numpy array): Array of segment times from the spectrogram.
        smoothed_signal (numpy array): The smoothed time-domain signal.
        peaks (numpy array): Indices of detected peaks.
        detected_keywords (list): List of tuples containing the timestamp and detected keyword.

    Returns:
        None: Displays a plot.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(times, smoothed_signal, label='Smoothed Signal')
    plt.plot(times[peaks], smoothed_signal[peaks], 'rx', label='Peaks')  # Mark peaks with red 'x'
    for keyword in detected_keywords:
        plt.annotate(keyword[1], (keyword[0], smoothed_signal[peaks[np.where(times == keyword[0])[0][0]]]),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('Detected Peaks and Keywords')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
