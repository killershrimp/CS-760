import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

import util
import decision_tree

inp_f = "data/Dbig.txt"

train, test = util.split_data(util.read_data(inp_f), 0.8192)

matroyshka = [train[:32], train[:128], train[:512], train[:2048], train]

learning_curve = []

for data in matroyshka:
    # P2.6
    DT = decision_tree.make_subtree(data)
    nodes = decision_tree.count_nodes(DT)
    learning_curve.append([nodes, 1 - decision_tree.test_subtree(test, DT)])
    print(len(data), nodes)
    max_x = np.max(test[:,0])
    max_y = np.max(test[:,1])
    min_x = np.min(test[:,0])
    min_y = np.min(test[:,1])
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)

    train_error = []
    for x_ in x:
        for y_ in y:
            train_error.append([x_, y_, decision_tree.classify(DT, [x_, y_])])

    util.plot_data(train_error)

    # sklearn
    # SK_DT = tree.DecisionTreeClassifier()
    # SK_DT = SK_DT.fit(data[:,:2], data[:,-1])
    # correct = 0
    # for i in test:
    #     if SK_DT.predict(i[:2].reshape(1,-1)) == i[-1]:
    #         correct += 1
    # error = 1 - correct / len(test)
    # nodes = SK_DT.tree_.node_count
    # print(len(data), nodes, error)
    # learning_curve.append([nodes, error])

learning_curve = np.array(learning_curve)
plt.xlabel("Number of Nodes in DT")
plt.ylabel("Classification Error")
plt.ylim([0, 1])
plt.title("Decision Tree Learning Curve")
plt.scatter(learning_curve[:,0], learning_curve[:,1])
plt.plot()
plt.show()
