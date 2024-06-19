
from predict_all_scenes import predict
from utils.data_utils import load_scenes_melspect_splitted 

if __name__ == "__main__":
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

    scenes_train, scenes_val, scenes_test = load_scenes_melspect_splitted()

    cost_per_scene = predict(model_file_path, scenes_test, stride, distances, thresholds)

    print(f"Cost per scene: {cost_per_scene}")
