import random
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    all_data = []
    with open(filename, "r") as file:
        line = file.readline()
        while line.rstrip() != "":
            all_data.append(np.array([float(i) for i in line.split(" ")]))
            line = file.readline()
    return np.array(all_data)


def split_data(dataset, train_size=0.5):
    random.shuffle(dataset)
    train = dataset[:int(len(dataset) * train_size)]
    test = dataset[int(len(dataset) * train_size):]
    return train, test


def plot_data(dataset):
    c1 = [[], []]
    c2 = [[], []]
    for i in dataset:
        if i[-1] == 0:
            c1[0].append(i[0])
            c1[1].append(i[1])
        else:
            c2[0].append(i[0])
            c2[1].append(i[1])
    plt.title("Data Distribution with " + str(len(dataset)) + " Data Points")
    plt.scatter(c1[0], c1[1], c="green", label="Label = 0")
    plt.scatter(c2[0], c2[1], c="red", label="Label = 1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

