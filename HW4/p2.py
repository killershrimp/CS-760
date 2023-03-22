import numpy as np

import util
from NaiveBayes import NaiveBayes


train_files_n = []
for lang in util.k_L:
    l = []
    for i in range(10):
        l.append(lang + str(i) + ".txt")
    train_files_n.append(l)

d1_train = []
for l in train_files_n:
    for filen in l[1:]:
        d1_train.append(util.read_train_file(filen))
NB1 = NaiveBayes(d1_train, 0.5)

# problem 1
print("\nProblem 1:")
for i in range(len(util.k_L)):
    lang = util.k_L[i]
    smoothed = NB1.smooth_simple(d1_train, lang)
    print("Language " + lang + ":", smoothed)

# problem 2
print("\nProblem 2:")
for i in range(3):
    f_index = i * 10
    data = d1_train[f_index:f_index+10]
    a = [NB1.smooth_conditional(data, util.k_L[i], char) for char in util.k_S]
    print(["{0:0.3f}".format(i) for i in a])

# problem 3
print("\nProblem 3:")

# problem 4
print("\nProblem 4:")
d4 = util.read_test_file("e10.txt")
print("Bag of words for e10.txt:")
print(d4)

# problem 5, 6
print("\nProblem 5, 6:")
mll, pr = NB1.classify(d4)
print("Most likely language:", mll)

# problem 7
conf_mat = np.zeros((3,3))
for num in range(10, 20):
    for lang in range(len(util.k_L)):
        actual = util.k_L[lang]
        f7 = actual + str(num) + ".txt"
        data = util.read_test_file(f7)
        pred, prob = NB1.classify(data)
        conf_mat[np.where(util.k_L == pred)[0][0]][lang] += 1

for i in range(3):
    for j in range(3):
        print("&", conf_mat[i][j], end=" ")
    print()

