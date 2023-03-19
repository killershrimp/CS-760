import numpy as np

import util


class NaiveBayes:
    def __init__(self, d_train, alpha=0):
        self.alpha = alpha
        self.pr_y = []
        self.pr_x_given_y = []  # format: [ [class1, [char1, ...]] , [class2 ...]
        self.train(d_train)

    def smooth_conditional(self, data, language, char):
        """
        Smooths P(X=char | Y=language) to avoid extremes
        :param data: all data. struct: [ [filename1, [freq_!, freq_2, ...]] , [filename2, [freq_1....]] , ...]
        :param language: language to search for in the set: {"e", "j", "s"}
        :param char: character to search for in the set: ["a", ..., "z"] U " "
        :return: smoothed conditional probability
        """
        numer = 0
        denom = 0
        index = ord(char) - ord('a')
        if char == " ":
            index = 26

        for i in data:
            if i[0] == language:
                numer += i[1][index]
                denom += np.sum(i[1])

        return (numer + self.alpha) / (denom + len(util.k_S) * self.alpha)

    def smooth_simple(self, data, language):
        """
        Smooths P(Y=language) to avoid extremes
        :param data: all data. struct: [ [filename1, [char1, char2, ...]] , [filename2, [char....]] , ...]
        :param language: language to search for in the set: {"e", "j", "s"}
        :return: smoothed conditional probability
        """
        numer = 0
        for i in data:
            if i[0] == language:
                numer += 1
        return (numer + self.alpha) / (len(data) + len(util.k_L) * self.alpha)

    def smooth_xs(self, data):
        rv = []
        sum = np.sum(data)
        for i in range(len(data)):
            rv[i] = (data[i] + self.alpha) / (sum + len(util.k_L) * self.alpha)
        return rv

    def classify(self, data):
        # return likeliest_c, max_pred
        likeliest = None
        for lang in range(len(util.k_L)):
            prob = np.sum(np.multiply(self.pr_x_given_y[lang][1], data)) + self.pr_y[lang]
            # print(util.k_L[lang], "pr: ", prob)
            if likeliest is None or prob > likeliest[1]:
                likeliest = util.k_L[lang], prob
        return likeliest

    def train(self, d_train):
        for lang in util.k_L:
            self.pr_y.append(np.log(self.smooth_simple(d_train, lang)))

            rv = [lang, []]
            for c in util.k_S:
                rv[1].append(np.log(self.smooth_conditional(d_train, lang, c)))
            self.pr_x_given_y.append(rv)





