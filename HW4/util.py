import numpy as np

k_S = []
for i in range(26):
    k_S.append(chr(ord('a') + i))
k_S.append(" ")
k_S = np.array(k_S)

k_L = np.array(["e", "j", "s"])  # set of all possible classifications


def read_test_file(filename):
    bag = np.zeros(27)
    with open("languageID/" + filename, "r") as file:
        for line in file:
            for j in line:
                if j != "\n":
                    if j == " ":
                        bag[26] += 1
                    else:
                        bag[ord(j) - ord('a')] += 1
    return np.array(bag)


def read_train_file(filename):
    return filename[0], read_test_file(filename)

