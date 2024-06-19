import numpy as np
import torch
import sys 
sys.path.append('../libs')  # Update this path according to the location of your 'dataset' module
import utils.classes as classes
from utils.peakpicking_utils import pick_peaks, pick_peaks, sliding_window_prediction  # Import the peak picking functions
from models.classifier import AudioClassifierCNN
from utils.data_utils import load_scenes_melspect
from utils.speech_commands import find_speech_commands, scene_cost

import dataset

def predict(model_file_path, max_files, stride, distances, thresholds):

    # Load the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Load the scene data
    scenes_data = load_scenes_melspect(max_files=10)
    
    # Parameters
    window_size = 44 # Each window has 1.1 seconds

    scene_commands = {}

    for file_path, scene_data in scenes_data:
        # Predict keywords 
        scene_tensor = torch.tensor(np.array([scene_data]), dtype=torch.float32)

        all_predictions = sliding_window_prediction(
            model, scene_tensor, window_size=window_size, stride=stride
        )

        # Pick prediction peaks
        detected_peaks = pick_peaks(all_predictions, thresholds, distances)
        # detected_peaks = pick_peaks_threshold(all_predictions, thresholds=thresholds, distance=5)

        # Convert class indices to names
        detected_peaks_named = {classes.REVERSE_CLASSES[class_idx]: peaks for class_idx, peaks in detected_peaks.items()}

        scene_commands[file_path] = find_speech_commands(detected_peaks_named)

    true_scene_commands = dataset.speech_command_labels()

    total_cost = 0
   
    for scene_name, predicted_commands in scene_commands.items():
        total_cost += scene_cost(predicted_commands, true_scene_commands[scene_name])

    cost_per_scene = total_cost / len(scene_commands)

    return cost_per_scene

if __name__ == "__main__":

    # Unaugmented path
    model_file_path = r"saved_models/model_20240612_115643/model_lr_0.001.pth"

    # Augmented path
    model_file_path = r"saved_models/model_20240616_143956/model_lr_0.001.pth"

    # Unoptimized peak picking values:
    thresholds = {
        "uninteresting": 0.3,
        "staubsauger": 0.3,
        "alarm": 0.3,
        "lüftung": 0.3,
        "ofen": 0.3,
        "heizung": 0.3,
        "fernseher": 0.3,
        "licht": 0.3,
        "aus": 0.3,
        "an": 0.3,
        "radio": 0.3,
    }
    distances = {
        "uninteresting": 5,
        "staubsauger": 5,
        "alarm": 5,
        "lüftung": 5,
        "ofen": 5,
        "heizung": 5,
        "fernseher": 5,
        "licht": 5,
        "aus": 5,
        "an": 5,
        "radio": 5,
    }
    stride = 11 # Advance by 1/4 of a window

    # Optimized peak picking values:
    
    cost_per_scene = predict(model_file_path, 10, stride, distances, thresholds)

    print(f"Cost per scene: {cost_per_scene}")
    
