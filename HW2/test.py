import numpy as np

from decision_tree import *


def read_data(filename):
    all_data = []
    with open(filename, "r") as file:
        line = file.readline()
        while line.rstrip() != "":
            all_data.append(np.array([float(i) for i in line.split(" ")]))
            line = file.readline()
    return np.array(all_data)


def split_data_evenly(dataset):
    train = dataset[:int(len(dataset) / 2)]
    test = dataset[int(len(dataset) / 2):]
    return train, test


inp_f = "data/D1.txt"

data = read_data(inp_f)
train, test = split_data_evenly(data)

DecisionTree = make_subtree(train)

accuracy = test_subtree(test, DecisionTree)
print("Accuracy:", accuracy)

