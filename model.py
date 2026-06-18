"""
model.py

for all your modelling needs
"""
# pylint: disable=C0116, C0103

from time import time as py_time

import torch
import matplotlib.pyplot as plt
from torch import nn, optim, no_grad

class LinearModel(nn.Module):
    """
    class to represent a very simple nn
    """
    def __init__(self, lr):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_reul_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        self.learning_rate = lr
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history = {'loss': [], 'training_time': 0.0}

    def forward(self, x):
        """
        Forward pass
        """
        x = self.flatten(x)
        logits = self.linear_reul_stack(x)
        return logits

    def train_model(self, train_loader, test_loader, num_epochs):

        # clear training history
        self.history['loss'] = []
        self.history['training_time'] = 0.0

        # For each epoch, run the training loop, then run the test (really validation) loop
        start_time = py_time()
        for t in range(num_epochs):
            print(f'EPOCH {t+1}\n-----------------------------')
            self.train_loop(train_loader)
            test_loss = self.test_loop(test_loader)

            self.history['loss'].append(test_loss)

        end_time = py_time()
        self.history['training_time'] = end_time - start_time
        print('DONE!')


    def train_loop(self, dataloader):
        size = len(dataloader.dataset)
        self.train()

        # train_loop
        for batch, (X,y) in enumerate(dataloader):
            # compute loss
            pred = self.forward(X)
            loss = self.loss_fn(pred, y)

            # backpropgation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * dataloader.batch_size + len(X)
                print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

    def test_loop(self, dataloader) -> float:
        self.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with no_grad():
            for X, y in dataloader:
                pred = self.forward(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_loss

    def display_training_history(self, output_file='./fig.png'):
        plt.plot(range(len(self.history['loss'])), self.history['loss'])
        plt.title(f'Training Time: {self.history['training_time']:>2f} Seconds')
        plt.savefig(output_file)
