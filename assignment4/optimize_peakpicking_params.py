from predict_all_scenes import predict
from utils.data_utils import load_scenes_melspect_splitted
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

def flatten_params(stride, distances, thresholds):
        return {
            "t_uninteresting": thresholds["uninteresting"],
            "t_staubsauger": thresholds["staubsauger"],
            "t_alarm": thresholds["alarm"],
            "t_lüftung": thresholds["lüftung"],
            "t_ofen": thresholds["ofen"],
            "t_heizung": thresholds["heizung"],
            "t_fernseher": thresholds["fernseher"],
            "t_licht": thresholds["licht"],
            "t_radio": thresholds["radio"],
            "t_aus": thresholds["aus"],
            "t_an": thresholds["an"],
            "d_uninteresting": distances["uninteresting"],
            "d_staubsauger": distances["staubsauger"],
            "d_alarm": distances["alarm"],
            "d_lüftung": distances["lüftung"],
            "d_ofen": distances["ofen"],
            "d_heizung": distances["heizung"],
            "d_fernseher": distances["fernseher"],
            "d_licht": distances["licht"],
            "d_radio": distances["radio"],
            "d_aus": distances["aus"],
            "d_an": distances["an"],
            "stride": stride
        }

def unflatten_params(args):

        thresholds = {
            "uninteresting": args["t_uninteresting"],
            "staubsauger": args["t_staubsauger"],
            "alarm": args["t_alarm"],
            "lüftung": args["t_lüftung"],
            "ofen": args["t_ofen"],
            "heizung": args["t_heizung"],
            "fernseher": args["t_fernseher"],
            "licht": args["t_licht"],
            "aus": args["t_aus"],
            "an": args["t_an"],
            "radio": args["t_radio"],
        }
        distances = {
            "uninteresting": int(args["d_uninteresting"]),
            "staubsauger": int(args["d_staubsauger"]),
            "alarm": int(args["d_alarm"]),
            "lüftung": int(args["d_lüftung"]),
            "ofen": int(args["d_ofen"]),
            "heizung": int(args["d_heizung"]),
            "fernseher": int(args["d_fernseher"]),
            "licht": int(args["d_licht"]),
            "aus": int(args["d_aus"]),
            "an": int(args["d_an"]),
            "radio": int(args["d_radio"]),
        }

        stride = int(args["stride"])
    
        return stride, distances, thresholds

if __name__ == "__main__":
    # Augmented path
    model_file_path = r"saved_models/model_20240616_143956/model_lr_0.001.pth"

    # Unoptimized peak picking values:
    thresholds = {
        "uninteresting": (0,1),
        "staubsauger": (0,1),
        "alarm": (0,1),
        "lüftung": (0,1),
        "ofen": (0,1),
        "heizung": (0,1),
        "fernseher": (0,1),
        "licht": (0,1),
        "aus": (0,1),
        "an": (0,1),
        "radio": (0,1),
    }

    distances = {
        "uninteresting": (3,44),
        "staubsauger": (3,44),
        "alarm": (3,44),
        "lüftung": (3,44),
        "ofen": (3,44),
        "heizung": (3,44),
        "fernseher": (3,44),
        "licht": (3,44),
        "aus": (3,44),
        "an": (3,44),
        "radio": (3,44),
    }
    stride = (1,44) # Advance by 1/4 of a window

    scenes_train, scenes_val, scenes_test = load_scenes_melspect_splitted()

    def f(**args):
        stride, distances, thresholds = unflatten_params(args)
        cost = predict(model_file_path, scenes_val , stride, distances, thresholds)
        # Want to maximize negative cost
        return -cost
    
    optimizer = BayesianOptimization(
        f=f,
        pbounds=flatten_params(stride, distances, thresholds),
        random_state=1,
    )

    # logger = JSONLogger(path="./logs.log")
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    # Load progress, uncomment if you have up to date logs.log and want to resume optimization
    # load_logs(optimizer, logs=["./logs.log"]);
    
    optimizer.maximize(n_iter=100, init_points=10)
    print(optimizer.max)
