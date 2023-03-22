import numpy as np


# def gen_d_softmax(z):
#     rv = np.zeros((len(z), len(z)))
#     softmaxes = self.softmax(z)
#     for i in range(len(z)):
#         for j in range(i, len(z)):
#             if i == j:
#                 rv[i][j] = softmaxes[i] * (1 - softmaxes[i])
#             else:
#                 temp = -softmaxes[i] * softmaxes[j]
#                 rv[i][j] = temp
#                 rv[j][i] = temp
#     return rv
#
# def gen_d_crossentropy(y_estimate, y_real):
#     return - np.divide(y_real, y_estimate)
import torch


def gen_d_crossentropy_comp_softmax(y_est, y_real):
    return y_est - y_real


def gen_d_sigmoid(x):
    s = sigmoid(x)
    return s * (np.ones(np.shape(x)) - s)


def gen_d_vecbymat(W, a):
    """
    :return d(Wa)/da where W is a matrix and a is a vector
    """
    # return np.repeat(a, np.shape(W)[0], axis=1)
    return W

k_clip_max = 19
def clip(i):
    """
    Prevent overflow in sigmoid
    :param i: original argument for sigmoid
    :return: clipped value (no sigmoid applied)
    """
    return np.clip(i, -k_clip_max, k_clip_max)


def sigmoid(x):
    return torch.sigmoid(torch.from_numpy(x)).numpy()


def _sigmoid(x):
    x = clip(x)
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


def softmax(x):
    return torch.softmax(torch.from_numpy(x), dim=-1).numpy()


def loss_crossentropy(y, y_pred):
    return - np.log(y_pred[y])


def get_one_hot(index, size):
    rv = np.zeros(size)
    rv[index] = 1
    return rv
