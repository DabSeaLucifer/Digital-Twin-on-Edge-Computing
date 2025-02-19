import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define Dataset Path
DATASET_PATH = "./data"

def get_cifar10_datasets():
    """ Load CIFAR-10 dataset for FL training """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_data_loader(dataset, batch_size=32):
    """ Return a DataLoader for a given dataset """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
