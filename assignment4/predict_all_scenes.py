import numpy as np
import torch
import sys 
sys.path.append('../libs')  # Update this path according to the location of your 'dataset' module
import utils.classes as classes
from utils.peakpicking_utils import pick_peaks, sliding_window_prediction  # Import the peak picking functions
from models.classifier import AudioClassifierCNN
from utils.data_utils import load_scenes_melspect
from utils.speech_commands import find_speech_commands, scene_cost

import dataset


def main():

    # Define the model file path
    model_file_path = r"saved_models/model_20240608_141753/model_lr_0.01.pth"

    # Load the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Load the scene data
    scenes_data = load_scenes_melspect(max_files=3)
    
    # Parameters
    sample_rate = 16000  # Example sample rate
    window_size = 44
    stride = 11

    scene_commands = {}

    for file_path, scene_data in scenes_data:
        # Predict keywords 
        scene_tensor = torch.tensor(np.array([scene_data]), dtype=torch.float32)

        all_predictions, threshold_predictions = sliding_window_prediction(
            model, scene_tensor, window_size=window_size, stride=stride, threshold=0.30
        )

        # Pick prediction peaks
        detected_peaks = pick_peaks(all_predictions, sample_rate, stride, height=0.3, distance=5)  

        # Convert class indices to names
        detected_peaks_named = {classes.REVERSE_CLASSES[class_idx]: peaks for class_idx, peaks in detected_peaks.items()}

        scene_commands[file_path] = find_speech_commands(detected_peaks_named)

    true_scene_commands = dataset.speech_command_labels()

    total_cost = 0

    for scene_name, predicted_commands in scene_commands.items():
        total_cost += scene_cost(predicted_commands, true_scene_commands[scene_name])

    print(f"Cost per scene: {total_cost / len(scene_commands)}")

if __name__ == "__main__":
    main()
