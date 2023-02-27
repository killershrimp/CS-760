# logistic regression

import numpy as np

import util


class LRModel:
    def __init__(self, learn_rate=10, epochs=1000):
        self.learn_rate = learn_rate
        self.weights = None
        self.epochs = epochs
        np.seterr(over='ignore')

    def predict_proba(self, data):
        """
        Predict probability output is 1
        :param weights: theta
        :param data: x
        :return: likelihood of desired class
        """
        return 1 / (1 + np.exp(- self.weights @ data.flatten()))

    def fit(self, train_feat, train_values):
        """
        Trains the Logistic Regression model
        :param train_feat: training set where each sample is a row of features + label (at the end)
        :param train_values: correct labels corresponding to training set
        """
        feat_n = len(train_feat[0])  # each emails row also contains y
        self.weights = np.zeros(feat_n)
        for i in range(self.epochs):
            for i in range(len(train_feat)):
                x = np.array(train_feat[i])
                self.weights -= x * self.learn_rate * (self.predict_proba(x) - train_values[i])

    def save_weights(self, filename="LRWeights.csv"):
        util.save_weights(self.weights, filename)

    def load_weights(self, filename="LRWeights.csv"):
        util.get_weights(filename)

    def print_weights(self):
        print(self.weights)
