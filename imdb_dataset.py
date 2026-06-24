"""
imdb_dataset.py

Class and Function definitions for 
"""

from torch.utils.data import Dataset

# pylint: disable=C0116, C0103, C0115

class IMDBFromScratch(Dataset):

    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        return 0

    @classmethod
    def load_imdb(cls, directory):
        return None, None
