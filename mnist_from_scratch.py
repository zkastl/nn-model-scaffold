"""
mnist_from_scratch.py

helper functions and classes for reading mnist data directly from file
"""
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

def read_train_images(filepath:str):
    """
    Reads the images from the mnist set
    """
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number for images: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def read_train_labels(filepath:str):
    """
    Reads the labels from the MNIST set
    """
    with open(filepath, 'rb') as f:
        magic, _ = struct.unpack('>II', f.read(8))
        assert magic == 2049, f'Invalid magic number for labels: {magic}'
        return np.frombuffer(f.read(), dtype=np.uint8)


class MNISTFromScratch(Dataset):
    """
    MNISTFromScratch
    Class to load the data and labels in a pytorch dataset format
    """
    def __init__(self, image_path, label_path, transform=None) -> None:
        super().__init__()
        self.images = read_train_images(image_path)
        self.labels = read_train_labels(label_path)
        self.transform = transform

    def __getitem__(self, index):
        _image = self.images[index].astype(np.float32) / 255.0
        _image = torch.tensor(_image).unsqueeze(0)

        if self.transform:
            _image = self.transform(_image)

        _label = int(self.labels[index])
        return _image, _label

    def __len__(self):
        return len(self.labels)
