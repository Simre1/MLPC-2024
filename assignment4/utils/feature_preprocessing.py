#!/usr/bin/env python3

import csv
import numpy as np
from pathlib import Path
import re


def load_data(filepath: str="") -> str:
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
            print(f"Loading {len(data_files)} files from {filepath} successfully!")
        except FileNotFoundError as e:
            print(e)

    return sorted(data_files)


def data_slicing(data, feature_selector: str="melspect", sampling_rate: int=16, frame_size: int= 44, hop_size: int=1):
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

    segment_labels = extract_labels_from_filename(data.stem)

    data = np.load(data)

    #########################################################################################
    # Hardcoded index slicing
    #########################################################################################
    if feature_selector == "":
        selected_data = data
    elif feature_selector == "bandwidth":
        selected_data = data[0]
    elif feature_selector == "centroid":
        selected_data = data[1]
    elif feature_selector == "contrast":
        selected_data = data[2:8]
    elif feature_selector == "energy":
        selected_data = data[9]
    elif feature_selector == "flatness":
        selected_data = data[10]
    elif feature_selector == "flux":
        selected_data = data[11]
    elif feature_selector == "melspect":
        selected_data = data[12:76]
    elif feature_selector == "mfcc":
        selected_data = data[77:108]
    elif feature_selector == "mfcc_d":
        selected_data = data[108:140]
    elif feature_selector == "mfcc_d2":
        selected_data = data[140:172]
    elif feature_selector == "power":
        selected_data = data[172]
    elif feature_selector == "yin":
        selected_data = data[173]
    elif feature_selector == "zcr":
        selected_data = data[174]
    else:
        raise ValueError(f"Please select one available feature or leave empty for all features!")
    
    #########################################################################################
    # Setup window function
    #########################################################################################

    num_frames = selected_data.shape[0]
    num_segments = (num_frames - frame_size) // hop_size + 1
    segments = [selected_data[i:i + frame_size] for i in range(0, num_frames - frame_size + 1, hop_size)]

    return np.array(segments), np.array(segment_labels)


def extract_labels_from_filename(filename) -> list:
    """Use regex to extract the part after "true_" or "false_"""
    
    match = re.search(r"_(true|false)_(.*)", filename)
    if match:
        labels_str = match.group(2)
        labels = labels_str.split("_")
        return labels
    return []


def main():
    features_filepath = "./development_scenes_npy/development_scenes/"
    labels_filepath = "./"
    raw_data = load_data(features_filepath)

    # for data in raw_data:
    #     print(data)

    split_data = data_slicing(raw_data[2])

    for data2 in split_data:
        print(data2)

    pass




if __name__ == "__main__":
    main()