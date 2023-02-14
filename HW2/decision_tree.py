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

    def __init__(self, split):
        """
        :param split: 1x2 vector: [feature index, split threshold]
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

    def get_split(self):
        return self.split

    def eval_split(self, value):
        return eval_split(value, self.split)


class Leaf:
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
    :param split: 1x2 vector: [feature index, split threshold]
    :return: if split is above/below threshold
    """
    return value[int(split[0])] >= split[1]


def p_y(d, y):
    """
    Returns the frequency of some output value in the dataset
    """
    return np.count_nonzero(d == y) / len(d)


def p_split(d, split):
    """
    Returns the frequency of split
    :param d: n full-feature inputs
    :param split: 1x2 vector: [feature index, split threshold]
    """
    count = 0
    for i in d:
        if eval_split(i, split):
            count += 1
    return count / len(d)


def p_conditional(d, split, val2, not_c=True):
    """
    :param d: all input instances in the form [one specific feature, output]
    :param split: 1x2 vector: [feature index, split threshold]
    :param val2: desired output value
    :return: frequency of value2 given split in the entire dataset
    """
    count = 0
    denom = 0
    for i in d:
        if not (eval_split(i, split) ^ not_c):
            denom += 1
            if i[-1] == val2:
                count += 1
    if denom == 0:
        return 0
    return count / denom


def entropy_y(all_data):
    """
    :param all_data: all data
    :return: entropy of label
    """
    rv = 0
    ys = np.unique(all_data[:, -1])
    for y in ys:
        p = p_y(all_data[:, -1], y)
        if p == 0:
            return 0
        rv -= p * math.log2(p)
    return rv


def entropy_given_x(all_data, split, not_c=True):
    """
    :param all_data: entire dataset
    :param split: one specific feature split [feature index, split threshold]
    :param not_c: want to use complement of split
    :return: entropy given feature value
    """
    h_given_x = 0
    for y in np.unique(all_data[:, -1]):
        denom = p_conditional(all_data, split, y, not_c)
        if denom != 0:
            h_given_x += denom * math.log2(denom)
    return h_given_x


def total_cond_entropy(all_data, split):
    """
    :param all_data: entire dataset
    :param split: possible split as 1x2 vector with form
     [feature index, split threshold]
    :return H(Y | X = x_i). entropy given a split
    """

    psplit = p_split(all_data, split)

    a = -entropy_given_x(all_data, split)
    b = -entropy_given_x(all_data, split, False)
    rv = a * psplit + b * (1-psplit)
    return rv


def info_gain(all_data, split):
    """
    determines info gain on some feature
    :param all_data: all data
    :param split: split to evaluate info gain of (1x2 vector: [feature index, split threshold])
    :return: info gain given split on feature
    """
    rv = entropy_y(all_data) - total_cond_entropy(all_data, split)
    return rv


def gain_ratio(all_data, split):
    a = p_split(all_data, split)
    b = 1 - a
    denom = 0
    if a != 0:
        denom += -math.log2(a) * a
    if b != 0:
        denom += -math.log2(b) * b
    if denom == 0:
        return 0
    return info_gain(all_data, split) / denom


def determine_candidate_splits(data):
    """
    Determine all candidate splits for given dataset. Only consider splits on single features
    :return: ith jth element = ith feature, jth possible split
    """
    num_features = len(data[0]) - 1  # max number of features
    all_splits = []
    for feature in range(num_features):
        # feature + result
        features = data[:, feature]
        features = np.sort(features)
        for i in range(len(features)-1):
            if features[i] != features[i+1]:
                # split between training set features
                # all_splits.append(np.array([feature, (features[i][0] + features[i + 1][0]) / 2, want_higher]))

                # split on features in training set
                split = features[i+1]
                all_splits.append(np.array([feature, split]))
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
    for split in c:
        if gain_ratio(data, split) != 0:
            all_gain_zero = False
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
            counts[1] += 1
    if counts[0] > counts[1]:
        return 0
    else:  # when no majority default to 1
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
    for split in splits[1:]:
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
    return np.array(r1), np.array(r2)


def make_subtree(data):
    c = determine_candidate_splits(data)
    if should_stop(data, c):
        return Leaf(get_label(data))
    else:
        d1, d2, split = find_best_split(data, c)
        n = Node(split)
        n.set_left(make_subtree(d1))
        n.set_right(make_subtree(d2))
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


def print_subtree(node, index=0):
    index_s = "(Index " + str(index) + ")"
    if type(node) == Leaf:
        print(index_s, "Leaf with label: ", node.get(), "\\\\")
        return
    feature_s = str(int(node.get_split()[0])) + ")"
    print(index_s, "Split (feature", feature_s, " greater than or eq. to", node.get_split()[1],
          "(see index " + str(2 * index + 1) + ");",
          "less than, (see index " + str(2 * index + 2) + ")", "\\\\")

    print_subtree(node.get_left(), 2 * index + 1)
    print_subtree(node.get_right(), 2 * index + 2)


def count_nodes(root):
    if type(root) == Leaf:
        return 1
    return count_nodes(root.get_left()) + count_nodes(root.get_right()) + 1
