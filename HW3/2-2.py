import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import util

n_feat = 3000
n_folds = 5
ks = [1,3,5,7,10]
# ks = [1]

words, emails = util.read_emails()
np.random.shuffle(emails)

for k in ks:
    print("K:", k)
    nn = KNeighborsClassifier(n_neighbors=k, p=2)
    avg_acc = util.cv_five(nn, emails)[0]

    # for 2-4
    plt.scatter(k, avg_acc)
    print("\t\tavg_acc:", avg_acc)

plt.margins(0.3)
plt.xlim(0,10.2)
plt.ylim(-0.3,1.3)
plt.xlabel("No. Nearest Neighbors")
plt.ylabel("Average Accuracy Across " + str(n_folds) + "-fold CV")
plt.show()





