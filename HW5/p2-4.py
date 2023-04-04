from util import *

fn_2d = "data/data2D.csv"
fn_1000d = "data/data1000D.csv"

X = read_file(fn_2d)
X_big = read_file(fn_1000d)


def plot_og():
    plt.scatter(X[:, 0], X[:, 1], color='g', label="Original Data")


print("Average Reconstruction Errors:")

k_use_1000d = False

dataset = X_big if k_use_1000d else X
k_d = 30 if k_use_1000d else 1


# plot_og()
rep_d, params, rep_D = PCA_buggy(dataset, k_d)
# plt.scatter(rep_D[:,0], rep_D[:,1], color='b', label="Reconstructed Data (Buggy)")
# plt.legend()
# plt.title("PCA Buggy Reconstruction vs Original")
# plt.savefig("p2-4_PCAbuggy.png")
# plt.clf()
print("PCA Buggy:", recon_error(dataset, rep_D))

a, b, rep_D_dm = PCA_demeaned(dataset, k_d)
# plot_og()
# plt.scatter(rep_D_dm[:,0], rep_D_dm[:,1], color='b', label="Reconstructed Data (Demeaned)")
# plt.legend()
# plt.title("PCA Demeaned Reconstruction vs Original")
# plt.savefig("p2-4_PCAdemeaned.png")
# plt.clf()
print("PCA Demeaned:", recon_error(dataset, rep_D_dm))

a, b, rep_D_norm = PCA_norm(dataset, k_d)
# plot_og()
# plt.scatter(rep_D_norm[:,0], rep_D_norm[:,1], color='b', label="Reconstructed Data (Normalized)")
# plt.legend()
# plt.title("PCA Normalized Reconstruction vs Original")
# plt.savefig("p2-4_PCAnormed.png")
# plt.clf()
print("PCA Normalized:", recon_error(dataset, rep_D_norm))

a, b, rep_D_dro = DRO(dataset, k_d)
# plot_og()
# plt.scatter(rep_D_dro[:,0], rep_D_dro[:,1], color='b', label="Reconstructed Data (DRO)")
# plt.legend()
# plt.title("DRO Reconstruction vs Original")
# plt.savefig("p2-4_DRO.png")
# plt.clf()
print("DRO:", recon_error(dataset, rep_D_dro))

