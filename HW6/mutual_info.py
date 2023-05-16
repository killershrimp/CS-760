import numpy as np


def mi(data, X, Y):
    rv = 0
    p_x = 0
    p_y = 0
    for row in data:
        if row[X] == 1:
            p_x += row[-1]

        if row[Y] == 1:
            p_y += row[-1]

    for x in (0, 1):
        for y in (0, 1):
            pr_xy = 0
            for row in data:
                is_x = row[X] == x
                is_y = row[Y] == y

                if is_x and is_y:
                    pr_xy += row[-1]

            rv += pr_xy * np.log2(pr_xy / (p_x * p_y))
    return rv


data = []
with open("p4.txt") as file:
    line = file.readline().rstrip()
    while line != "":
        data.append(line.split(","))
        line = file.readline().rstrip()

data = np.array(data, dtype=np.double)
data[:, -1] /= 100

print("I(X,Y) =", mi(data, 0, 1))
print("I(X,Z) =", mi(data, 0, 2))
print("I(Y,Z) =", mi(data, 1, 2))


