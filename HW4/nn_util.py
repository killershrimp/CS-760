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

def gen_d_crossentropy_comp_softmax(y_est, y_real):
    return y_est - y_real


def gen_d_sigmoid(x):
    rv = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        s = 1 / (1 + np.exp(-x[i]))
        rv[i][i] = s * (1 - s)
    return rv


def gen_d_matrix(W, a):
    """
    :return d(Wa)/dW where W is a matrix and a is a vector
    """
    return np.repeat(a, np.shape(W)[0], axis=1)


def sigmoid(x):
    fn = np.vectorize(lambda i: 1 / (1 + np.exp(-i)))
    return fn(x)


def softmax(x):
    return np.divide(np.exp(x), np.sum(np.exp(x)))


def loss_crossentropy(y, y_pred):
    return - np.sum(np.multiply(y, np.log(y_pred)))


def get_one_hot(index, size):
    rv = np.zeros(size)
    rv[index] = 1
    return rv
