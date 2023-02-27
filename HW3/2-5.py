import numpy as np

import LogisticRegression
import util
from plot_roc import plot_roc
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


words, emails = util.read_emails()
np.random.shuffle(emails)

test = emails[:int(len(emails)/3)]
train = emails[int(len(emails)/3):]

X = train[:,:-1]
y = train[:,-1]

KNN = KNeighborsClassifier(p=2, n_neighbors=5)
KNN.fit(X, y)
LR = LogisticRegression.LRModel()
LR.fit(X, y)

print("Done training!")

NN_fn = "roc_data/ROC_5NN.txt"
LR_fn = "roc_data/ROC_LR.txt"

with open(NN_fn, "w") as f_KNN:
    with open(LR_fn, "w") as f_LR:
        for i in test:
            x = i[:-1].reshape(1,-1)
            y = i[-1]
            f_KNN.write(str(KNN.predict_proba(x).flatten()[1]) + " " + str(y) + "\n")
            f_LR.write(str(LR.predict_proba(x)) + " " + str(y) + "\n")

x_KNN, y_KNN = plot_roc(NN_fn)
x_LR, y_LR = plot_roc(LR_fn)

plt.plot(x_KNN, y_KNN, label="5NN", c='r')
plt.plot(x_LR, y_LR, label="LR", c='g')

plt.ylim(-0.1,1.1)
plt.xlim(-0.1,1.1)
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.title("ROC Curve for 5NN and Log. Regression")
plt.legend()
plt.show()


