# src/train.py

import os
import sys
from datetime import datetime
sys.path.append('../../libs')  # Update this path according to the location of your 'dataset' module

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Add the parent directory to the system path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloader import create_dataloaders
from utils.data_utils import load_data, preprocess_data, split_data, split_data_in_X_y
from utils.model_utils import train_model_with_lr, make_predictions_with_all_models
from utils.eval_utils import accuracy_fn
from utils.plot_utils import plot_confusion_matrix
from models.classifier import AudioClassifierCNN

import classes


def main():
    DEVELOPMENT_FILE = "../../Files/metadata/development.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and preprocess data
    X, y = load_data()
    X_new = preprocess_data(X)

    # Load the dataset
    data = pd.read_csv(DEVELOPMENT_FILE)
    train_data, test_data, dev_data = split_data(data)
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_data_in_X_y(X_new, y, train_data, test_data, dev_data)

    # Create dataloaders
    batch_size = 32
    train_dataloader, test_dataloader, dev_dataloader = create_dataloaders(X_train, X_test, X_dev, y_train, y_test,
                                                                           y_dev, batch_size)

    # Initialize the model
    model = AudioClassifierCNN(len(classes.CLASSES))
    model = model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam

    learning_rates = [1e-1, 1e-2]
    results = {}
    os.makedirs('saved_models', exist_ok=True)

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        model = AudioClassifierCNN(len(classes.CLASSES)).to(device)
        best_loss, train_time, trained_model = train_model_with_lr(lr, optimizer, train_dataloader, test_dataloader,
                                                                   model, loss_fn, accuracy_fn, device)

        # Create a directory with a timestamp to save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join('saved_models', f'model_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)

        # Save the model within the directory
        model_path = os.path.join(model_dir, f'model_lr_{lr}.pth')
        torch.save(trained_model.state_dict(), model_path)

        # Store the test loss along with other information

        results[lr] = {'best_loss': best_loss, 'train_time': train_time, 'model_path': model_path}

    # Save test losses to a CSV file
    test_losses_file = 'saved_models/models.csv'
    if os.path.exists(test_losses_file):
        # Append to existing file
        results_df = pd.read_csv(test_losses_file)
        new_results_df = pd.DataFrame(results).transpose()
        new_results_df['timestamp'] = datetime.now()
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    else:
        # Create new file
        results_df = pd.DataFrame(results).transpose()
        results_df['timestamp'] = datetime.now()

    # Save the DataFrame to a CSV file
    results_df.to_csv(test_losses_file, index=False)

    # Display results
    for lr, result in results.items():
        print(
            f"Learning Rate: {lr} | Best Test Loss: {result['best_loss']:.5f} | Training Time: {result['train_time']} | Model Path: {result['model_path']}")

    make_predictions_with_all_models(AudioClassifierCNN, results, len(classes.CLASSES), dev_dataloader, device)

    plot_classes = [key for key, _ in classes.CLASSES.items()]  # Extracting only the keys

    for lr, result in results.items():
        model_path = result['model_path']
        prediction_file = os.path.join(os.path.dirname(model_path), f"predictions.pth")
        saved_data = torch.load(prediction_file)
        y_true = saved_data['y_true'].numpy()
        y_pred = saved_data['y_pred'].numpy()

        plot_confusion_matrix(y_true, y_pred, plot_classes, prediction_file)

if __name__ == "__main__":
    main()
