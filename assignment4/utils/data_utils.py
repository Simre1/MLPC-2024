# src/utils/data_utils.py

import sys
import os
sys.path.append('../libs')  # Update this path according to the location of your 'dataset' module
import dataset
import classes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_data():
    # Load your dataset here
    # Assuming dataset.data() returns your numpy array and classes.label_to_class converts labels to classes
    X = dataset.data()
    y = np.array(list(map(classes.label_to_class, dataset.labels_array())))
    return X, y

def preprocess_data(X):
    # Preprocess the data, e.g., select specific features
    X_new = X[:, 12:76, :]
    return X_new

def split_data(data):
    # Get unique speaker IDs
    speaker_ids = data['speaker_id'].unique()

    # Split speaker IDs into train, test, and dev sets
    train_speaker_ids, test_dev_speaker_ids = train_test_split(speaker_ids, test_size=0.2, random_state=42)
    test_speaker_ids, dev_speaker_ids = train_test_split(test_dev_speaker_ids, test_size=0.5, random_state=42)

    # Filter data into train, test, and dev sets based on speaker IDs
    train_data = data[data['speaker_id'].isin(train_speaker_ids)]
    test_data = data[data['speaker_id'].isin(test_speaker_ids)]
    dev_data = data[data['speaker_id'].isin(dev_speaker_ids)]

    # Optionally, shuffle the train, test, and dev DataFrames
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    dev_data = dev_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return train_data, test_data, dev_data

def split_data_in_X_y(X, y, train_data, test_data, dev_data):
    # Get the indices of rows corresponding to speaker IDs in train, test, and dev sets
    # train_indices = oversampled_data['id'].values
    train_indices = train_data['id'].values
    test_indices = test_data['id'].values
    dev_indices = dev_data['id'].values

    # Split data in X and y data
    X_train = X[train_indices, :, :]
    X_test = X[test_indices, :, :]
    X_dev = X[dev_indices, :, :]

    y_train = y[train_indices]
    y_test = y[test_indices]
    y_dev = y[dev_indices]

    return X_train, X_test, X_dev, y_train, y_test, y_dev

def oversample_data(train_data, words_to_oversample):
    # Oversample specific words in the training data
    oversampled_data = pd.concat([resample(train_data[train_data['word'].isin(words_to_oversample)], 
                                           replace=True,  # With replacement
                                           n_samples=3000,  # Adjust oversampling size as needed
                                           random_state=42),  # Set random state for reproducibility
                                  train_data])
    # Shuffle the oversampled data
    oversampled_data = oversampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return oversampled_data

# Load scene files and return them as a list of np arrays
# Limit the amount of files with the max_files or leave it to load all scenes 
def load_scenes_melspect(max_files=0):
    directory = '../Files/development_scenes'
    extension = ".npy"
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    if max_files > 0:
        file_paths = file_paths[0:max_files]
    loaded_scenes = [(os.path.splitext(os.path.basename(file_path))[0], np.load(file_path)[12:76, :]) for file_path in file_paths]
    return loaded_scenes
