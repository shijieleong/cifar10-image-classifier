import torch

import torchvision
import torchvision.transforms as transforms

def load_data():
    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert data values from 0~255 to 0~1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Convert data values from 0~1 to -1~1
    ])

    # Import train and test data set
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Load data using torch
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

    return train_loader, test_loader