import numpy as np
import random
import librosa
from scipy.ndimage import shift as scipy_shift

def add_noise(data, noise_level=0.005):
    """Add random noise to the audio data."""
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

def shift_audio(data, shift_max=0.2):
    """Shift the audio data by a random fraction of the total length along the time axis."""
    shift_amount = int(shift_max * data.shape[1] * (2 * random.random() - 1))
    data_shifted = scipy_shift(data, (0, shift_amount), mode='nearest')
    return data_shifted

def stretch_audio(data, rate=1.1):
    """Stretch the audio data by a given rate along the time axis."""
    output_length = int(data.shape[1] / rate)
    stretched_data = np.zeros((data.shape[0], output_length))
    for i in range(data.shape[0]):
        interp_data = np.interp(np.arange(0, data.shape[1], rate), np.arange(0, len(data[i])), data[i])
        if len(interp_data) > output_length:
            interp_data = interp_data[:output_length]
        elif len(interp_data) < output_length:
            interp_data = np.pad(interp_data, (0, output_length - len(interp_data)), 'constant')
        stretched_data[i] = interp_data
    if stretched_data.shape[1] > data.shape[1]:
        stretched_data = stretched_data[:, :data.shape[1]]
    else:
        stretched_data = np.pad(stretched_data, ((0, 0), (0, data.shape[1] - stretched_data.shape[1])), "constant")
    return stretched_data


def pitch_shift_2d(data, n_steps):
#    """Apply pitch shifting to the 2D spectrogram data."""
#    # Convert spectrogram to time-domain signal
#    time_series = librosa.istft(data, n_fft=min((data.shape[0] - 1) * 2, 2048))
#    # Apply pitch shifting
#    time_series_shifted = librosa.effects.pitch_shift(time_series, sr=len(time_series), n_steps=n_steps)
#    # Convert back to spectrogram with hardcoded n_fft
#    n_fft = 1024
#    data_shifted = librosa.stft(time_series_shifted, n_fft=n_fft)
#    # Ensure the output shape matches the input shape
#    data_shifted = data_shifted[:data.shape[0], :data.shape[1]]
    return data

def time_stretch(data, rate):
    """Time stretch does not directly apply to 2D data and will be skipped."""
    return data

def add_reverb(data, reverb_level=0.5):
    """Add reverberation does not directly apply to 2D data and will be skipped."""
    return data

def equalize(data, low_freq, high_freq, gain_db):
    """Equalization does not directly apply to 2D data and will be skipped."""
    return data

def adjust_gain(data, gain_db):
    """Adjust the gain of the audio data."""
    return data * (10 ** (gain_db / 20))

class Augmenter:
    def __init__(self, noise_level=0.005, shift_max=0.2, stretch_rate=1.1, gain_db=5, augmentation_prob=0.5):
        self.noise_level = noise_level
        self.shift_max = shift_max
        self.stretch_rate = stretch_rate
        self.gain_db = gain_db
        self.augmentation_prob = augmentation_prob
        self.augmentations = [
            ('noisy', lambda data: add_noise(data, noise_level=self.noise_level)),
            ('shifted', lambda data: shift_audio(data, shift_max=self.shift_max)),
            ('stretched', lambda data: stretch_audio(data, rate=self.stretch_rate)),
            # ('pitch_shifted', lambda data: pitch_shift_2d(data, n_steps=2)),
            # ('time_stretched', lambda data: time_stretch(data, rate=1.1)), # Skipping as it doesn't apply to 2D
            # ('reverb', lambda data: add_reverb(data, reverb_level=0.5)), # Skipping as it doesn't apply to 2D
            # ('equalized', lambda data: equalize(data, low_freq=200, high_freq=3000, gain_db=5)), # Skipping as it doesn't apply to 2D
            ('gain_adjusted', lambda data: adjust_gain(data, gain_db=self.gain_db)),
        ]

    def augment(self, data):
        if random.random() < self.augmentation_prob:
            augmentation_name, augmentation_func = random.choice(self.augmentations)
            data = augmentation_func(data)
            if data.shape != (64, 44):
                print(f"Shape mismatch after augmentation: {data.shape}")
        return data