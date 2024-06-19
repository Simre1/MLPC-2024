# src/utils/model_utils.py

import os
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from timeit import default_timer as timer
from utils.eval_utils import accuracy_fn, print_train_time

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()  # Set the model to training mode
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        
        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()  # Accumulate the loss
        
        if y.ndim == 2:
            train_acc += accuracy_fn(y_true=y.argmax(dim=1), y_pred=y_pred.argmax(dim=1))  # Accumulate the accuracy
        else:
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Accumulate the accuracy

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

def test_step(data_loader, model, loss_fn, accuracy_fn, device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradient computation
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()  # Accumulate the loss
            if y.ndim == 2:
                test_acc += accuracy_fn(y_true=y.argmax(dim=1), y_pred=test_pred.argmax(dim=1))  # Accumulate the accuracy
            else:
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))  # Accumulate the accuracy

        # Calculate loss and accuracy per epoch and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

def train_model_with_lr(lr, optimizer_class, train_dataloader, test_dataloader, model, loss_fn, accuracy_fn, device):
    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    torch.manual_seed(42)

    # Measure time
    train_time_start_model = timer()

    delta = 1e-2
    patience = 5
    best_test_loss = float('inf')
    best_epoch = 0
    counter = 0

    # Train and test model 
    epochs = 20
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")

        train_loss, train_acc = train_step(data_loader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

        test_loss, test_acc = test_step(data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        # Check for early stopping
        if test_loss < best_test_loss - delta:
            best_test_loss = test_loss
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping. No improvement since epoch {best_epoch}.')
                break
    train_time_end_model = timer()
    total_train_time_model = print_train_time(start=train_time_start_model,
                                              end=train_time_end_model,
                                              device=device)
    return best_test_loss, total_train_time_model, model

def load_model(model_class, model_path, device, num_classes):
    model = model_class(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def make_predictions_with_all_models(model_class, model_paths, num_classes, dataloader, device):
    for lr, result in model_paths.items():
        model_path = result['model_path']
        print(f"\nMaking predictions with model trained with learning rate: {lr}")
        
        model = load_model(model_class, model_path, device, num_classes)
        model.eval()
        
        y_preds = []
        y_true = []
        
        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Making predictions"):
                # Send data and targets to target device
                X, y = X.to(device), y.to(device)
                # Do the forward pass
                y_logit = model(X)
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())
                y_true.append(y.cpu())
        
        # Concatenate list of predictions into a tensor
        y_pred_tensor = torch.cat(y_preds)
        y_true_tensor = torch.cat(y_true)
        
        # Calculate and print any metrics
        print(f"Predictions for model with learning rate {lr}:")
        accuracy = (y_pred_tensor == y_true_tensor).sum().item() / len(y_true_tensor)
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Save predictions
        save_path = os.path.join(os.path.dirname(model_path), f"predictions.pth")

        torch.save({'y_pred': y_pred_tensor, 'y_true': y_true_tensor}, save_path)
