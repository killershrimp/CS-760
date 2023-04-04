import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def PCA_buggy(X, d):
    """
    :param X: nxd data matrix
    :param d: dimension d of matrix X
    :return: d-D rep, est params, d-D rep in D-D
    """
    U, S, V = np.linalg.svd(X)
    S = np.diag(S)
    V_1 = V[:,:d]

    ddim = X @ V[:d].T

    return ddim, (S, V_1), ddim @ V[:d]


def demean(X,d):
    means = []
    for i in range(X.shape[1]):
        means.append(np.mean(X[:,i]))
    means = np.array(means)
    X_dm = np.copy(X)
    X_dm -= np.ones((X.shape[0], 1)) @ np.transpose(means[:,np.newaxis])
    return X_dm, means


def PCA_demeaned(X, d):
    """
    :param X: nxd data matrix
    :param d: dimension d of matrix X
    :return: d-D rep, est params, d-D rep in D-D
    """
    X_dm, means = demean(X,d)
    a, b, c = PCA_buggy(X_dm, d)
    # a += np.ones((d, 1)) @ np.transpose(means[:d][:,np.newaxis])
    c += np.ones((X.shape[0], 1)) @ np.transpose(means[:,np.newaxis])
    return a, b, c


def PCA_norm(X, d):
    """
    :param X: nxd data matrix
    :param d: dimension d of matrix X
    :return: d-D rep, est params, d-D rep in D-D
    """
    X_dm, means = demean(X, d)
    for i in range(X.shape[1]):
        X_dm[:,i] /= np.std(X[:,i])
    a, b, c = PCA_buggy(X_dm, d)
    for i in range(X.shape[1]):
        c[:,i] *= np.std(X[:,i])

    # use to find knee points
    # write_file("singular-values.txt", np.diagonal(b[0]))

    # a += np.ones((d, 1)) @ np.transpose(means[:d][:,np.newaxis])
    c += np.ones((X.shape[0], 1)) @ np.transpose(means[:,np.newaxis])
    return a, b, c


def DRO(X, d):
    """
    :param X: nxd data matrix
    :param d: dimension d of matrix X
    :return: d-D rep, est params, d-D rep in D-D
    """
    n = X.shape[0]
    b = np.transpose(X) @ np.ones((n,1)) / n
    X_new = X - np.ones((n,1)) @ np.transpose(b)

    U, S, V = np.linalg.svd(X_new)

    S_1 = np.copy(S)
    for i in range(d, len(S_1)):
        S_1[i] = 0
    A_1 = np.transpose(S_1 * V)

    return np.transpose(S[:d] * V[:d]), (A_1, b), U[:,:d][:,np.newaxis] @ S[:d] * V[:d] + np.ones((n,1)) @ b.T


def recon_error(a, b):
    return np.sum(np.square(np.linalg.norm(a - b))) / len(a)


def read_file(fname):
    X = []
    with open(fname, "r") as file:
        line = file.readline().strip()
        while line != "":
            X.append([float(i) for i in line.split(",")])
            line = file.readline().strip()
    return np.array(X)


def write_file(fname, data):
    with open(fname, "w") as file:
        for i in data:
            file.write(str(i) + "\n")
