from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

color_train = 'b'
train_x_0 = []
train_x_1 = []
train_y_0 = []
train_y_1 = []
with open("data/D2z.txt") as file:
    l = file.readline().rstrip().split(" ")
    while len(l) == 3:
        if l[-1] == '0':
            train_x_0.append([float(l[0]), float(l[1])])
            train_y_0.append(float(l[-1]))
        else:
            train_x_1.append([float(l[0]), float(l[1])])
            train_y_1.append(float(l[-1]))
        l = file.readline().rstrip().split(" ")

train_x_0 = np.array(train_x_0)
train_x_1 = np.array(train_x_1)
train_x = np.concatenate([train_x_0, train_x_1])
train_y = np.concatenate([train_y_0, train_y_1])

color_0 = 'r'
color_1 = 'g'

nn1 = KNeighborsClassifier(n_neighbors=1, p=2)
nn1.fit(train_x, train_y)

# Pr 2-1
x_12 = np.linspace(-2, 2, int(4 / 0.1))
y_12 = x_12.copy()
r_0 = []
r_1 = []
# r_0 = [nn1.predict(np.array([x_12[i], y_12[i]]).reshape(1, -1)) for i in range(len(x_12))]

for i in x_12:
    for j in y_12:
        if nn1.predict(np.array([i,j]).reshape(1, -1)) == 0:
            r_0.append([i,j])
        else:
            r_1.append([i,j])
r_0 = np.array(r_0)
r_1 = np.array(r_1)
plt.scatter(r_0[:,0], r_0[:,1], c=color_0, label='0 Value (Test)')
plt.scatter(r_1[:,0], r_1[:,1], c=color_1, label='1 Value (Test)')

plt.scatter(x=train_x_0[:,0], y=train_x_0[:,1], marker="+", label='0 Value (Train)')
plt.scatter(x=train_x_1[:,0], y=train_x_1[:,1], marker="x", label='1 Value (Train)')

plt.margins(0.3)
plt.title("Problem 2-1 Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig('2-1.png')
plt.show()


