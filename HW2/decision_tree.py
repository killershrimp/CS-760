# assumptions:
# - each item has 2 continuous features
# - class label is binary and encoded as y \in {0, 1}
# - data files are in plaintext with one labeled item per line, separated by whitespace (ex: "x1 x2 y1")
import math

import numpy as np

default_class = 1

# todo add pruning?


class Node:
    left = None
    right = None
    split = None
    is_leaf = False

    def __init__(self, split):
        """
        :param split: 1x3 vector: [feature index, split threshold, boolean is_up]
        """
        self.split = split

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def set_right(self, value):
        self.right = value

    def set_left(self, value):
        self.left = value

    def eval_split(self, value):
        return eval_split(value, self.split)


class Leaf():
    value = None

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def eval_split(value, split):
    """
    :param value: input vector
    :param split: 1x3 vector: [feature index, split threshold, boolean is_up]
    :return: if split is above/below threshold
    """
    if split[2]:
        return value[int(split[0])] > split[1]
    else:
        return value[int(split[0])] < split[1]


def p_y(d, y):
    """
    Returns the frequency of some output value in the dataset
    """
    return np.count_nonzero(d[:-1] == y)


def p_split(d, split):
    """
    Returns the frequency of split
    :param d: nx1 vector only containing specific feature
    :param split: 1x3 vector: [feature index, split threshold, boolean is_up]
    """
    count = 0
    for i in d:
        if eval_split(i, split):
            count += 1
    return count


def p_conditional(d, split, val2):
    """
    :param d: all input instances in the form [one specific feature, output]
    :param split: 1x3 vector: [feature index, split threshold, boolean is_up]
    :param val2: desired output value
    :return: frequency of split and value2 in the entire dataset
    """
    count = 0
    for i in d:
        if eval_split(i, split) and i[1] == val2:
            count += 1
    return count


def entropy_y(all_data):
    """
    :param all_data: all data
    :return: entropy of label
    """
    rv = 0
    ys = np.unique[all_data[:-1]]
    for y in ys:
        p = p_y(all_data, y)
        rv -= p * math.log2(p)
    return rv


def entropy_given_x(all_data, split):
    """
    :param all_data: entire dataset
    :param split: one specific feature split [feature index, split threshold, boolean is_up]
    :return: entropy given feature value
    """
    h_given_x = 0
    for y in np.unique(all_data[-1]):
        trimmed_d = np.concatenate(
            (all_data[:, int(split[0])].reshape(len(all_data), 1), all_data[:,-1].reshape(len(all_data), 1)),
            axis=1)  # #manscaped #ad

        a = p_conditional(trimmed_d, split, y)
        if a == 0:
            return 0
        h_given_x += p_y(trimmed_d, y) * math.log2(a)  # todo check signs
    return h_given_x


def total_cond_entropy(all_data, split):
    """
    :param all_data: entire dataset
    :param split: possible split as 1x3 vector with form
     [feature index, split threshold, boolean is_up]
    :return H(Y | X = x_i). entropy given a split
    """
    rv = 0
    h_given_x = entropy_given_x(all_data, split)
    rv -= p_split(all_data[:, split[0]], split) * h_given_x
    return rv


def info_gain(all_data, split):
    """
    determines info gain on some feature
    :param all_data: all data
    :param split: split to evaluate info gain of (1x3 vector: [feature index, split threshold, boolean is_up])
    :return: info gain given split on feature
    """
    return entropy_y(all_data) - total_cond_entropy(all_data, split)


def gain_ratio(all_data, split):
    # split data by parts and then sum todo generalize beyond binary?
    parts = [[], []]
    for entry in all_data:
        if entry[-1] == 0:
            parts[0].append(entry)
        else:
            parts[1].append(entry)

    sum_entropy_given_x = -entropy_given_x(np.array(parts[0]), split) - entropy_given_x(np.array(parts[1]), split)
    if sum_entropy_given_x == 0:
        return 0
    return info_gain(all_data, split) / sum_entropy_given_x


# todo integrate split entropy/gain ratio into split class and cache for later function use
def determine_candidate_splits(data):
    """
    Determine all candidate splits for given dataset. Only consider splits on single features
    :return: ith jth element = ith feature, jth possible split
    """
    num_features = len(data[0]) - 1  # max number of features
    all_splits = []
    for feature in range(num_features):
        feature_splits = []

        # feature + result
        features = np.concatenate(
            (data[:, feature].reshape(len(data), 1), data[:, -1].reshape(len(data), 1)),
            axis=1)

        np.sort(data, axis=0)  # sort by feature value
        for i in range(len(features)-1):
            if features[i][-1] != features[i + 1][-1]:
                # maybe generalize? only works for binary classification
                want_higher = features[i][-1] == 0  # T/F 1 is higher
                feature_splits.append(np.array([feature, (features[i][0] + features[i + 1][0]) / 2, want_higher]))
        all_splits.append(np.array(feature_splits))
    return np.array(all_splits)


def should_stop(data, c):
    """
    Stop if:
    1) node is empty (default to classifying as 1)
    2) entropy of all candidate splits is 0
    3) all splits have zero gain ratio
    :param data: full remaining dataset
    :param c: candidate splits at this point
    :return: T/F should stop
    """
    if len(data) == 0:
        return True

    # checks both if gain is 0 or entropy is 0
    all_gain_zero = True
    for feature in c:
        for split in feature:
            if gain_ratio(data, split) != 0:
                all_gain_zero = False
                break
        if not all_gain_zero:
            break
    if all_gain_zero:
        return True
    return False


def get_label(data):
    counts = [0, 0]
    for entry in data:
        if entry[-1] == 0:
            counts[0] += 1
        else:
            counts[1] +=1
    if counts[0] > counts[1]:
        return 0
    else:   # when no majority default to 1
        return default_class


def find_best_split(all_data, splits):
    """
    Find the best split given several splits
    :param all_data: full dataset
    :param splits: candidate splits [ [f1_split_1, f1_split 2...], [f2_split_1, ...], ... ]
    :return: (data partition 1, data part 2, best split by info gain ratio)
    """
    r_split = splits[0]
    r_gain = gain_ratio(all_data, r_split)
    for feature_splits in range(len(splits)):
        for split in splits[feature_splits]:
            n_gain = gain_ratio(all_data, split)
            if n_gain > r_gain:
                r_gain = n_gain
                r_split = split
    r1, r2 = split_by(all_data, r_split)
    return r1, r2, r_split


def split_by(data, split):
    """
    Split entries in dataset into 2.
    :return: features positively satisfying split, features negatively satisfying split
    """
    r1 = []
    r2 = []
    for i in data:
        if eval_split(i, split):
            r1.append(i)
        else:
            r2.append(i)
    return r1, r2


def make_subtree(data):
    c = determine_candidate_splits(data)
    if should_stop(data, c):
        return Leaf(get_label(data))
    else:
        d1, d2, split = find_best_split(data, c)
        n = Node(split)
        n.set_left(make_subtree(d1))
        n.set_right(make_subtree(d2))
        make_subtree(d2)
        return n


def classify(parent, inp):
    """
    Given a DT, classify a piece of input data
    :param parent: parent Node of DT
    :param inp: single input vector
    :return: output
    """
    if type(parent) == Leaf:
        return parent.get()

    left = parent.eval_split(inp)
    if left:
        return classify(parent.get_left(), inp)
    return classify(parent.get_right(), inp)


def test_subtree(data, parent):
    """
    test subtree
    :param data: test data
    :param parent: node of DT to start comparison at
    :return: percent accuracy [0. 1]
    """
    tests_passed = 0

    for test in data:
        if classify(parent, test) == test[-1]:
            tests_passed += 1

    return tests_passed / len(data)

