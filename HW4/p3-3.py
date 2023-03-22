import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

k_batch_size = 64
k_di = 28*28
k_d1 = 300
k_d2 = 200
k_d3 = 100
k_df = 10
k_epochs = 15
k_lr = 8

d_train = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
d_test = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

dl_train = DataLoader(d_train, k_batch_size)
dl_test = DataLoader(d_test, k_batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(k_di, k_d1, True),
            nn.Sigmoid(),
            nn.Linear(k_d1, k_d2, True),
            nn.Sigmoid(),
            nn.Linear(k_d2, k_df, True),
        )

        # zero weight init
        # self.stack.apply(self.zero_w_init)

        # # random weight init
        # self.stack.apply(self.random_w_init)

        self.loss_fn = nn.CrossEntropyLoss()

    def random_w_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.uniform_(layer.weight, -1, 1)

    def zero_w_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)

    def backward(self):
        return self.loss.backward()


model = NeuralNetwork().to("cpu")


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = model.loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, epochs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += model.loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    learn_curve_acc.append([epochs,correct])
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


optimizer = torch.optim.SGD(model.parameters(), lr=k_lr)

learn_curve_acc = []

for t in range(k_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dl_train, model, optimizer)
    test_loop(dl_test, model, t)
print("Done!")

plt.title("Learning Curve for NN with Batch Size " + str(k_batch_size))
learn_curve_acc = np.array(learn_curve_acc)
plt.plot(learn_curve_acc[:,0], learn_curve_acc[:,1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
print("\n\n", learn_curve_acc)
print("\n\n", (np.ones(len(learn_curve_acc)) - learn_curve_acc[:,1]))

torch.save(model.state_dict(), 'pt_MNIST_model_weights.pth')

