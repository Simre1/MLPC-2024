# src/utils/plot_utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(y_true, y_pred, classes, prediction_file):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix for {}".format(prediction_file))

    base_filename = os.path.basename(prediction_file)
    filename = os.path.splitext(base_filename)[0]
    save_path = os.path.join(os.path.dirname(prediction_file), f"{filename}_confusion_matrix.jpg")

    plt.savefig(save_path)
    # plt.show()
