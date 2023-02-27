import matplotlib.pyplot as plt
import numpy as np

import LogisticRegression
import util

learn_rate = 0.1
LR = LogisticRegression.LRModel(learn_rate)

words, emails = util.read_emails()

np.random.shuffle(emails)

util.cv_five(LR, emails)


# FOR TUNING LEARN RATE/EPOCHS
# acc = []
# prec = []
# rec = []

# for j in range(-1, 2):
#     print("j:", j)
#     for i in range(1, 5):
#         epochs = int(10**i)
#         print("epochs:", epochs)
#         model = LogisticRegression.LRModel(learn_rate=10**j, epochs=epochs)
#         avg_acc, avg_prec, avg_rec = util.cv_five(model, emails)
#         acc.append(avg_acc)
#         prec.append(avg_prec)
#         rec.append(avg_rec)
#
# plt.scatter(range(12), acc, label="Accuracy", c='r')
# plt.scatter(range(12), prec, label="Precision", c='g', marker='+')
# plt.scatter(range(12), rec, label="Recall", c='b', marker="x")
#
# plt.ylabel("Average Accuracy, Precision, Recall with 5-fold CV")
# plt.xlabel("Log (Base 10) of Step Size")
# plt.ylim(-0.3, 1.3)
# plt.legend()
# plt.show()



