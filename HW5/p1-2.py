import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

k_num_clusters = 3
k_one_sample = 100


def accuracy(model, data):
    guesses = model.predict(data)
    values, counts = np.unique(guesses, return_counts=True, axis=0)
    mode = np.argmax(counts, axis=0)
    return counts[mode] / np.sum(counts)


def objective(model, means, data):
    pred_means = means[model.predict(data)]
    dists = np.linalg.norm(pred_means - data, axis=0)
    return np.sum(np.square(dists))


sigmas = [0.5, 1, 2, 4, 8]
means = np.array([[-1, -1], [1, -1], [0, 1]])
accuracies = []
distances = []

for sigma in sigmas:
    a_gauss = np.random.multivariate_normal(means[0], sigma * np.array([[2, 0.5],[0.5,1]]), k_one_sample)
    b_gauss = np.random.multivariate_normal(means[1], sigma * np.array([[1, -0.5],[-0.5,2]]), k_one_sample)
    c_gauss = np.random.multivariate_normal(means[2], sigma * np.array([[1, 0],[0,2]]), k_one_sample)

    X = np.concatenate((a_gauss, b_gauss, c_gauss), axis=0)
    np.random.shuffle(X)

    kms = KMeans(n_clusters=k_num_clusters, n_init=10, init="k-means++", algorithm="lloyd").fit(X)
    gmm = GaussianMixture(n_components=k_num_clusters, n_init=10).fit(X)

    accuracies.append(np.array([accuracy(kms, a_gauss) + accuracy(kms, b_gauss) + accuracy(kms, c_gauss),
                      accuracy(gmm, a_gauss) + accuracy(gmm, b_gauss) + accuracy(gmm, c_gauss)])/3)
    distances.append(np.array([objective(kms, kms.cluster_centers_, X),
                     objective(gmm, gmm.means_, X)]))
accuracies = np.array(accuracies)
distances = np.array(distances)
plt.plot(sigmas, accuracies[:,0], label="Accuracies", color="g")
plt.xlabel("Sigma")
plt.ylabel("Accuracy")
plt.title("KMeans Accuracy vs Sigmas")
plt.savefig("p1-2-1a.png")
plt.clf()

plt.plot(sigmas, distances[:,0], label="Distance Errors", color="r")
plt.xlabel("Sigma")
plt.ylabel("Distance Errors")
plt.title("KMeans Objectives vs Sigmas")
plt.savefig("p1-2-1b.png")
plt.clf()

plt.plot(sigmas, accuracies[:,1], label="Accuracies", color="g")
plt.xlabel("Sigma")
plt.ylabel("Accuracy")
plt.title("GMM Accuracy vs Sigmas")
plt.savefig("p1-2-2a.png")
plt.clf()

plt.plot(sigmas, distances[:,1], label="Distance Errors", color="r")
plt.xlabel("Sigma")
plt.ylabel("Distance Errors")
plt.title("GMM Objectives vs Sigmas")
plt.savefig("p1-2-2b.png")
plt.clf()
