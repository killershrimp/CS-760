import numpy as np
from torch import nn        # only used for data processing
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from nn_util import *

import matplotlib.pyplot as plt


k_batch_size = 32
k_di = 28*28
k_d1 = 300
k_d2 = 200
k_d3 = 100
k_df = 10
k_epochs = 5
k_lr = 1e-4

d_train = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
d_test = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

dl_train = DataLoader(d_train, batch_size=k_batch_size)
dl_test = DataLoader(d_test, batch_size=k_batch_size)


class NeuralNetwork:
    def __init__(self,
                 lr,
                 w1=np.random.choice(np.linspace(-1,1),size=(k_d1, k_di)),
                 w2=np.random.choice(np.linspace(-1,1),size=(k_d2, k_d1)),
                 w3=np.random.choice(np.linspace(-1,1),size=(k_df, k_d2))):
        self.W1 = w1
        self.W2 = w2
        self.W3 = w3
        self.learn_rate = lr

        self.W1x = None
        self.sig_1 = None
        self.W2x = None
        self.sig_2 = None
        self.W3x = None

    def forward(self, X):
        rv = []

        self.W1x = []
        self.sig_1 = []
        self.W2x = []
        self.sig_2 = []
        self.W3x = []
        for batch in range(np.shape(X)[1]):
            self.W1x.append(self.W1 @ X[:,batch])
            self.sig_1.append(sigmoid(self.W1x[batch]))
            self.W2x.append(self.W2 @ self.sig_1[batch])
            self.sig_2.append(sigmoid(self.W2x[batch]))
            self.W3x.append(self.W3 @ self.sig_2[batch])
            rv.append(softmax(self.W3x[batch]))
        return np.array(rv)

    def backward(self, X_all, y_est_all, y_real_all):
        dW3 = np.zeros(np.shape(self.W3))
        dW2 = np.zeros(np.shape(self.W2))
        dW1 = np.zeros(np.shape(self.W1))
        for batch in range(len(y_real_all)):
            X = X_all[:,batch][:,np.newaxis]
            y_est = y_est_all[batch]
            y_real = y_real_all[batch]

            y = np.transpose((y_est - y_real)[:,np.newaxis])
            z_3 = self.W1x[batch]
            sig_z3 = self.sig_1[batch][:,np.newaxis]
            z_2 = self.W2x[batch]
            z_1 = self.sig_2[batch]

            # calculate gradients
            # todo use dag?
            d_loss_sm = gen_d_crossentropy_comp_softmax(y_est, y_real)
            dW3 += d_loss_sm[:,np.newaxis] @ np.transpose(z_1[:,np.newaxis])

            d_sigmoid_z2 = gen_d_sigmoid(z_2)
            base_2 = y @ (self.W3 * d_sigmoid_z2)
            dW2 += np.transpose(sig_z3 @ base_2)

            dW1 += np.transpose(X @ base_2 @ self.W2 * gen_d_sigmoid(z_3))

        # update weights
        scaled_lr = self.learn_rate / len(y_real_all)
        self.W3 -= scaled_lr * dW3
        self.W2 -= scaled_lr * dW2
        self.W1 -= scaled_lr * dW1


def train_loop(dataloader, model):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        # nn.Flatten(X)
        X = torch.reshape(X, (k_di, len(y))).numpy().reshape(k_di, -1)
        y = y.numpy()
        y_pred = model.forward(X)
        loss = np.average([loss_crossentropy(y[i], y_pred[i]) for i in range(len(y_pred))])
        model.backward(X, y_pred, y)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * np.shape(X)[1]
            print(f"{batch}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, epochs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for X,y in dataloader:
        # nn.Flatten(X)
        X = torch.reshape(X, (k_di, len(y))).numpy().reshape(k_di, -1)
        y = y.numpy()
        y_pred = model.forward(X)
        test_loss += np.average([loss_crossentropy(y[i], y_pred[i]) for i in range(len(y_pred))])
        print(y_pred[-1])
        correct += np.sum([np.argmax(y_pred[i]) == y[i] for i in range(len(y_pred))])

    test_loss /= num_batches
    correct /= size
    learn_curve_acc.append([epochs,correct])
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model = NeuralNetwork(k_lr)
learn_curve_acc = []

for t in range(k_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dl_train, model)
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