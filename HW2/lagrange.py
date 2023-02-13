import scipy
import numpy as np
import math


def MS(y, exp_y):
    return math.sqrt(y**2 + exp_y**2)


a = 0
b = 4 * math.pi

n_points = 20
p_train = 0.5       # % total points used for training

mean = 0
std_dev = 0.1

x = np.linspace(a, b, num=n_points)
y = [math.sin(i) for i in x]

z = np.concatenate([np.reshape(x, [len(x), 1]), np.reshape(y, [len(y), 1])], axis=1)
np.random.shuffle(z)

train = z[:int(p_train * n_points)]
test = z[int(p_train * n_points):]

for i in range(len(train)):
    train[i] += np.random.normal(mean, std_dev)

f = scipy.interpolate.lagrange(train[:,0], train[:,1])

error = 0
for i in test:
    error += MS(i[-1], f(i[0]))
error /= len(test)

print("Error:", error)

