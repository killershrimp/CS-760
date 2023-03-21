from torch import nn        # only used for data processing
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from nn_util import *


k_batch_size = 64
k_di = 28*28
k_d1 = 300
k_d2 = 200
k_d3 = 100
k_df = 10
k_epochs = 5
k_lr = 5

d_train = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
d_test = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

dl_train = DataLoader(d_train, k_batch_size)
dl_test = DataLoader(d_test, k_batch_size)


class NeuralNetwork:
    def __init__(self,
                 lr,
                 w1=np.random.randn(k_d1, k_di),
                 w2=np.random.randn(k_d2, k_d1),
                 w3=np.random.randn(k_df, k_d2)):
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
        self.W1x = self.W1 @ X
        self.sig_1 = sigmoid(self.W1x)
        self.W2x = self.W2 @ self.sig_1
        self.sig_2 = sigmoid(self.W2x)
        self.W3x = self.W3 @ self.sig_2
        return softmax(self.W3x)

    def backward(self, X_all, y_est_all, y_real_all):
        dW3 = np.zeros(np.shape(self.W3))
        dW2 = np.zeros(np.shape(self.W2))
        dW1 = np.zeros(np.shape(self.W1))
        for batch in range(k_batch_size):
            X = X_all[:,batch]
            y_est = y_est_all[:,batch]
            y_real = y_real_all[batch]

            z_3 = self.W1x[:,batch]
            sig_z3 = self.sig_1[:,batch]
            z_2 = self.W2x[:,batch]
            z_1 = self.sig_2[:,batch]

            # calculate gradients
            # todo use dag?
            d_loss_sm = gen_d_crossentropy_comp_softmax(y_est, y_real)
            dW3 += d_loss_sm[:,np.newaxis] @ np.transpose(z_1[:,np.newaxis])

            # print(batch, "w3 update")

            d_sigmoid_z2 = gen_d_sigmoid(z_2)
            base_2 = d_sigmoid_z2 @ gen_d_matrix(self.W3, z_1[:,np.newaxis]) @ d_loss_sm[:,np.newaxis]
            mat_2 = np.zeros(np.shape(self.W2))
            for i in range(np.shape(self.W2)[0]):
                for j in range(np.shape(self.W2)[1]):
                    row = np.array([(0 if i != j else sig_z3[i])for i in range(k_d2)])
                    mat_2[i][j] = row @ base_2
            dW2 += mat_2

            # print(batch, "w2 update")

            x = gen_d_matrix(self.W2, sigmoid(z_3)[:,np.newaxis])
            base_1 = gen_d_sigmoid(z_3) @ x @ base_2
            mat_1 = np.zeros(np.shape(self.W1))
            for i in range(np.shape(self.W1)[0]):
                for j in range(np.shape(self.W1)[1]):
                    row = np.array([(0 if i != j else X[i])for i in range(k_d1)])
                    mat_1[i][j] = row @ base_1
            dW1 += mat_1
            print(batch, "w1 update")

        # update weights
        scaled_lr = self.learn_rate / k_batch_size
        self.W3 -= scaled_lr * dW3
        self.W2 -= scaled_lr * dW2
        self.W1 -= scaled_lr * dW1


def train_loop(dataloader, model):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        nn.Flatten(X)
        X = X.numpy().reshape(k_di, k_batch_size)
        y = y.numpy()
        y_pred = model.forward(X)
        loss = loss_crossentropy(y, y_pred)
        model.backward(X, y_pred, y)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"{batch}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    for X,y in dataloader:
        X = X.numpy().flatten()
        y = y.numpy()
        y_pred = model.forward(X)
        test_loss += loss_crossentropy(y, y_pred)
        correct += (y_pred.argmax(1) == y).type(np.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model = NeuralNetwork(k_lr)

for t in range(k_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dl_train, model)
    test_loop(dl_test, model)
print("Done!")