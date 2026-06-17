"""
main.py

Entry Point for Application
"""
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

TRAIN_IMAGE_FILEPATH = '/home/zak/data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILEPATH = '/home/zak/data/mnist/train-labels.idx1-ubyte'

def read_train_images(filepath:str):
    """
    Reads the images from the mnist training set
    """
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number for images: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)
    
def read_train_labels(filepath:str):
    """
    Reads the labels from the MNIST training set
    """
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f'Invalid magic number for labels: {magic}'
        return np.frombuffer(f.read(), dtype=np.uint8)

read_train_images(filepath=TRAIN_IMAGE_FILEPATH)
read_train_labels(filepath=TRAIN_LABEL_FILEPATH)
