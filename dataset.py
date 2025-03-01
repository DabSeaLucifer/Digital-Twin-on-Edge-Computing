import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_PATH = "./data"

def get_cifar10_datasets():
    """ Load CIFAR-10 dataset for FL training """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))

    return train_dataset, test_dataset

def get_data_loader(dataset, batch_size=32):
    """ Return a DataLoader for a given dataset """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
