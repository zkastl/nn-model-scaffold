"""
main.py

Entry Point for Application
"""
import uuid
from torch.utils.data import DataLoader

from mnist_from_scratch import MNISTFromScratch
from imdb_dataset import IMDBFromScratch
from model import ConvolutionalModel, LinearModel

###########################
# Dataset Preparation
TRAIN_IMAGE_FILEPATH    = '/home/zak/data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILEPATH    = '/home/zak/data/mnist/train-labels.idx1-ubyte'
TEST_IMAGE_FILEPATH     = '/home/zak/data/mnist/t10k-images.idx3-ubyte'
TEST_LABEL_FILEPATH     = '/home/zak/data/mnist/t10k-labels.idx1-ubyte'

TRAIN_TEXT_DIR          = '/home/zak/data/imdb/train'
TEST_TEXT_DIR           = '/home/zak/data/imdb/test'

BATCH_SIZE = 64
NUM_EPOCHS = 40
UNIQUE_ID = uuid.uuid7().int

# IMAGE DATASET
train_dataset = MNISTFromScratch(image_path=TRAIN_IMAGE_FILEPATH, label_path=TRAIN_LABEL_FILEPATH)
test_dataset = MNISTFromScratch(image_path=TEST_IMAGE_FILEPATH, label_path=TEST_LABEL_FILEPATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
image_train, label_train = next(iter(train_loader))
image_test, label_test = next(iter(test_loader))

# TEXT DATASET
train_texts, train_labels = IMDBFromScratch.load_imdb(TRAIN_TEXT_DIR)


# MODELS

# Train and save model 1
model = LinearModel(lr=0.01)
model.train_model(train_loader, test_loader, num_epochs=NUM_EPOCHS)
model.display_training_history(output_file=f'{UNIQUE_ID}_linear_model.png')

# Train and save model 2
model = ConvolutionalModel(lr=0.01)
model.train_model(train_loader, test_loader, num_epochs=NUM_EPOCHS)
model.display_training_history(output_file=f'{UNIQUE_ID}_conv_model.png')
