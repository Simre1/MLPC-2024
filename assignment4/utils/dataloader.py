# src/utils/dataloader.py

import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(X_train, X_test, X_dev, y_train, y_test, y_dev, batch_size=32):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(dim=1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(dim=1)
    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32).unsqueeze(dim=1)
    y_train_tensor = torch.tensor(y_train)
    y_test_tensor = torch.tensor(y_test)
    y_dev_tensor = torch.tensor(y_dev)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    dev_dataset = TensorDataset(X_dev_tensor, y_dev_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    return train_dataloader, test_dataloader, dev_dataloader
