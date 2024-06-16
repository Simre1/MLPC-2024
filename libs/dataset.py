import numpy as np
import random
import csv
import pathlib


# Opinionated paths for loading the files
DATASET_DIR = "Files/development_numpy/development.npy"
IDX_TO_FEATURE_NAME_FILE = "Files/metadata/idx_to_feature_name.csv"
DEVELOPMENT_FILE = "Files/metadata/development.csv"

# Cached feature indices and labels for easy access
CACHED_FEATURE_INDICES = None
CACHED_LABELS = None
CACHED_DATA = None
PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()

def data():
    """
    Load the data and return it.

    Returns:
        np.ndarray: The loaded dataset as a NumPy array.
    """

    global CACHED_DATA
    if CACHED_DATA is not None:
        return CACHED_DATA


    CACHED_DATA = np.load(PROJECT_PATH / DATASET_DIR)
    return CACHED_DATA

def features_indices():
    """
    Retrieves the feature indices from the CSV file and stores them in a dictionary.
    
    Returns:
        feature_names (dict): A dictionary where the keys are feature names and the values are
            tuples containing the start and end indices of the feature.
    """
    global CACHED_FEATURE_INDICES
    if CACHED_FEATURE_INDICES is not None:
        return CACHED_FEATURE_INDICES

    csv_file = PROJECT_PATH / IDX_TO_FEATURE_NAME_FILE
    feature_name_idx_dict = {}
    with open(csv_file, newline='', encoding='utf8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            index = int(row[0])
            feature_name_parts = row[1].split('_')
            feature_name = '_'.join(feature_name_parts[:-1]).lower()  # Extract feature name without the number suffix and convert to lowercase
            if feature_name not in feature_name_idx_dict:
                feature_name_idx_dict[feature_name] = (index, index)
            else:
                start_idx, _ = feature_name_idx_dict[feature_name]
                feature_name_idx_dict[feature_name] = (start_idx, index)

    CACHED_FEATURE_INDICES = feature_name_idx_dict
    return CACHED_FEATURE_INDICES

def get_size_of_feature(feature_name):
    """
    Calculate the size of a feature based for the given feature name.

    Returns:
        int: The size of the feature.
    """
    feature_name_idx_dict = features_indices()
    return feature_name_idx_dict[feature_name][1] - feature_name_idx_dict[feature_name][0] + 1

def get_feature_indices(feature_name):
    """
    Returns the indices of a feature name in the features_indices dictionary.
        
    Returns:
        list: A list of indices corresponding to the feature name.
        
    Raises:
        ValueError: If the feature name is not found in the indices dictionary.
    """
    feature_name_idx_dict = features_indices()
    lower_feature_name = feature_name.lower()

    if lower_feature_name in feature_name_idx_dict:
        range = feature_name_idx_dict[lower_feature_name]
        return np.arange(range[0], range[1]+1)
    else:
        raise ValueError(f"Feature name {feature_name} not found in the indices dictionary.")

def print_feature_names():
    print("The feature names are:")
    print("")
    for key, value in features_indices().items():
        print(f"{key}")

def replace_special_characters(word):
    # Replace special characters with ASCII equivalents
    word = word.replace('ö', 'oe').replace('ä', 'ae').replace('ü', 'ue').replace('ß', 'ss')
    return word

def labels_array():
    csv_file = PROJECT_PATH / DEVELOPMENT_FILE
    labels_list = []

    with open(csv_file, newline='', encoding='utf8') as csvfile:  # Specify UTF-8 encoding
        csv_reader = csv.DictReader(csvfile)
        csv_reader.line_num
        for row in csv_reader:
            word = row['word'].lower()
            # file_index = int(row['id'])
            # filename_parts = row['filename'].split('/')
            # filename = '/'.join(filename_parts[-2:])
            labels_list.append(word)
    
    return np.array(labels_list)

def labels():
    """
    Retrieves the labels from the CSV file

    Returns:
        dict: A dictionary mapping each label to a list of samples which have that label.

    """
    global CACHED_LABELS
    if CACHED_LABELS is not None:
        return CACHED_LABELS

    csv_file = PROJECT_PATH / DEVELOPMENT_FILE
    class_to_idx_list_dict = {}
    with open(csv_file, newline='', encoding='utf8') as csvfile:  # Specify UTF-8 encoding
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            word = row['word'].lower()
            file_index = int(row['id'])
            filename_parts = row['filename'].split('/')
            filename = '/'.join(filename_parts[-2:])
            class_to_idx_list_dict.setdefault(word, []).append(
                dict(id=file_index, filename=filename, speaker_id=row['speaker_id'])
            )

    CACHED_LABELS = class_to_idx_list_dict
    return CACHED_LABELS

def print_label_lengths():
    print("The size of the different samples of each class is:")
    print("")
    for class_label, sample_ids in labels().items():
        print(f"{class_label}: {len(sample_ids)}")

def speech_command_labels():
    csv_file = PROJECT_PATH / "Files/development_scene_annotations.csv"

    commands_for_name = {}
    with open(csv_file, newline='', encoding='utf8') as csvfile:  # Specify UTF-8 encoding
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            name = row['filename']
            [object, action] = row['command'].lower().split(" ")
            start = float(row['start'])
            end = float(row['end'])
            commands_for_name.setdefault(name, []).append(
                (object, action, start, end)
            )
    
    return commands_for_name
