import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
from config import Config  # Add this import

def get_transform(is_train=True):
    config = Config()  # Create an instance of Config
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])