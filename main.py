"""
main.py

Entry Point for Application
"""
from torch.utils.data import DataLoader

from mnist_from_scratch import MNISTFromScratch
from model import LinearModel

###########################
# Dataset Preparation
TRAIN_IMAGE_FILEPATH = '/home/zak/data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILEPATH = '/home/zak/data/mnist/train-labels.idx1-ubyte'
TEST_IMAGE_FILEPATH = '/home/zak/data/mnist/t10k-images.idx3-ubyte'
TEST_LABEL_FILEPATH = '/home/zak/data/mnist/t10k-labels.idx1-ubyte'
BATCH_SIZE = 64
NUM_EPOCHS = 40

train_dataset = MNISTFromScratch(image_path=TRAIN_IMAGE_FILEPATH, label_path=TRAIN_LABEL_FILEPATH)
test_dataset = MNISTFromScratch(image_path=TEST_IMAGE_FILEPATH, label_path=TEST_LABEL_FILEPATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

image_train, label_train = next(iter(train_loader))
image_test, label_test = next(iter(test_loader))

assert len(train_dataset) == 60000
assert len(test_dataset) == 10000
###########################

###########################
# Model Preparation

model = LinearModel(lr=0.01)
model.train_model(train_loader, test_loader, num_epochs=NUM_EPOCHS)

# print out training chart
model.display_training_history(output_file='./fig.png')
