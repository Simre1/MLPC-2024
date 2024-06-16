#!/usr/bin/env python

import sys
sys.path.append('../../libs')  # Update this path according to the location of your 'dataset' module

import csv
import numpy as np
from pathlib import Path
import re

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import dataset
import preprocessing
import classes

####################################################################################################
# GET LIST OF FEATURE FILES
####################################################################################################

def get_numeric_prefix(file):
    match = re.match(r"(\d+)_", file.name)
    return int(match.group(1)) if match else float('inf')

####################################################################################################
# GET NUMBER TO SORT BECAUSE PYTHON IS STUPID
####################################################################################################

def get_feature_files(filepath: str="") -> str:
    """Reads all files.

    Arguments:
        filepath: str
            Path to directory containing all files.
    Returns:
        data_files: list
            List of all files in the directory.
    """

    dir_data = Path(filepath)
    if dir_data.exists():
        # print(f"DIR DATA: {dir_data}")
        try:
            data_files = list(dir_data.glob("*.npy"))
            data_files.sort(key=get_numeric_prefix)
            print(f"Loading {len(data_files)} files from {filepath} successfully!")

        except FileNotFoundError as e:
            print(f"Wrong directory: {e}")

    return data_files


####################################################################################################
# GET FEATURE DICTIONARY
####################################################################################################

def get_feature_dict(filepath, delimiter=","):
    feature_dictionary = {}
    filepath = Path(filepath)
    with filepath.open("r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        next(reader)  # Skip the header
        for row in reader:
            index = int(row[0])
            name = row[1].rsplit("_", 1)[0]
            # Check if the name exists in the dictionary
            if name not in feature_dictionary:
                feature_dictionary[name] = []
            # Append the index to the list for this name
            feature_dictionary[name].append(index)
    
    return feature_dictionary

####################################################################################################
# GET FEATURE INDICES
####################################################################################################

def get_features_indices(feature_dictionary, *feature_names):
    feature_indices = []
    for feature in feature_names:
        if feature in feature_dictionary:
            feature_indices.extend(feature_dictionary[feature])
    return feature_indices

####################################################################################################
# GET LABEL NAMES
####################################################################################################  

def extract_labels_from_filename(filename) -> list:
    """Use regex to extract the part after "true_" or "false_"""
    
    match = re.search(r"_(true|false)_(.*)", filename)
    if match:
        labels_str = match.group(2)
        labels = labels_str.split("_")
        return labels
    return []

####################################################################################################
# GET NUMPY FILE AND FILENAME
####################################################################################################

def load_filterd_data(numpy_file=None, index=None, feature_indices=None):

    if index is None:

        file = np.load(numpy_file)
        filtered_data = file[feature_indices, :]

        return filtered_data, Path(numpy_file)
    else:

        file = np.load(numpy_file[index])
        filtered_data = file[feature_indices, :]

        return filtered_data, Path(numpy_file[index])

####################################################################################################
# SLICE DATA
####################################################################################################  

def slice_data(feature_file, sampling_rate: int=16, frame_size: int=44, hop_size: int=1):
    """Slices data in smaller segments for further processing.

    Arguments:
        data: ndarray
            Input features of the development set.
        feature_selector: str
            Select either individual features only or the entire dataset.
        sampling_rate: int
            Sampling rate in kHz.
        frame_size: int
            Number of frames in a window.
        hop_size: int
            Number of frames to move the window (stride).

    Returns:
        sliced_data: ndarray
            Individually stacked feature frames.
    """
    file = feature_file[0]
    path = feature_file[1]
    #print(f"File: {path}")

    segment_labels = extract_labels_from_filename(path.stem)
    #print("======================")
    #print(segment_labels)
    #print("#################")
    #print(file)

    #########################################################################################
    # Setup window function
    #########################################################################################

    num_frames = file.shape[0]
    #print(type(file))
    #for i in range(file.shape[0]):
    #    print(f"Element {i+1}: {file[i].shape}")
    print(f"File Shape: {file.shape}")
    #print(f"File1 Shape: {file[1].shape}")
    #print(f"File2 Shape: {file[2].shape}")
    #print(f"File3 Shape: {file[3].shape}")
    num_segments = (num_frames - frame_size) // hop_size + 1
    #print(f"Segments: {num_segments}")
    segments = [file[i:i + frame_size] for i in range(0, num_frames - frame_size + 1, hop_size)]

    return np.array(segments), np.array(segment_labels)


####################################################################################################
# MAIN
####################################################################################################


def main():
    features_filepath = "../../Files/development_scenes_npy/development_scenes/"
    labels_filepath = "../../Files/metadata/idx_to_feature_name.csv"
    num = 10

    feature_files = get_feature_files(features_filepath)
    print(f"File {num}: {feature_files[num]}")
    feature_dict = get_feature_dict(labels_filepath)
    feature_indices = get_features_indices(feature_dict, "melspect")

    #file_2_raw = np.load(feature_files[2])
    #file_2_raw = load_numpy(feature_files, 2)
    #file_2_filtered = file_2_raw[feature_indices, :]
    #print(f"Length Feature Index: {len(feature_indices)}")
    #for file in feature_files:
    #    print(file)

    file_2_filtered_data, file_2_filtered_path = load_filterd_data(feature_files, index=num, feature_indices=feature_indices)

    #print(f"SHAPE: {file_2_filtered_data.shape}")
    # for data in raw_data:
    #     print(data)

    file_2_filtered = [file_2_filtered_data, file_2_filtered_path]

    sliced_data = slice_data(file_2_filtered)

    #print(f"Sliced Data 0: {sliced_data[0]}")
    print(f"Sliced Data 1: {sliced_data[1]}")

    #for data2 in split_data:
    #    print(data2)




if __name__ == "__main__":
    main()