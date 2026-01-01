# src/dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size=128):
    """
    Define basic transforms for mammography images.

    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                       
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),    
    ])


def get_dataloaders(data_dir="data", img_size=128, batch_size=16, num_workers=4):
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(img_size)
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_transforms(img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, train_dataset.classes
