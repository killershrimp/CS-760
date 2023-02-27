import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def read_emails():
    words = []
    emails = []  # each row is an email, each column 1-3000 is a feature, last column is label

    with open("data/emails.csv") as file:
        words = file.readline().rstrip().split(" ")[1].split(",")
        l = file.readline().rstrip().split(" ")
        while len(l) == 2:
            emails.append(l[1].split(",")[1:])  # every row starts with "Email (email #),#,#...."
            l = file.readline().rstrip().split(" ")

    emails = np.array(emails).astype('float64')
    return words, emails


def test_model(model, test_x, test_y, conf_thresh=0.5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(test_x)):
        prob = model.predict_proba(test_x[i].reshape(1,-1)).flatten()
        if type(model) == KNeighborsClassifier:
            prob = prob[1]
        r = prob > conf_thresh
        if r:
            if int(test_y[i]) == 1:
                tp += 1
            else:
                fp += 1
        else:
            if int(test_y[i]) == 1:
                fn += 1
            else:
                tn += 1
    tot = len(test_x)

    acc = (tp + tn) / tot
    prec = (tp / (tp + fp)) if (tp + fp) != 0 else 0
    rec = (tp / (tp + fn)) if (tp + fn) != 0 else 0
    return acc, prec, rec


def save_weights(theta, filename="weights.csv"):
    """
    Save a vector of weights
    :param theta: weights to save
    :param filename: file to save weights to
    """
    with open(filename, 'a') as f:
        for i in theta:
            if i != theta[-1]:
                f.write(str(i) + ",")
            else:
                f.write(str(i))


def get_weights(filename):
    """
    Retrieve saved weights from a file
    :param filename: name of file weights are stored in
    :return: weights as a 1D vector
    """
    with open(filename, 'r') as f:
        return np.array(f.readline().strip().split(","))


def cv_five(model, emails):
    """
    Run five-fold cross validation
    """
    n_feat = len(emails[0])-1
    avg_acc = 0
    avg_prec = 0
    avg_rec = 0

    print("\\begin{tabular}{|| c c c c ||} \\hline")
    print("Fold No. & Accuracy & Precision & Recall \\\\ \\hline \\hline")
    for i in range(5):
        test_s = 1000 * i
        test_e = 1000 * (i + 1)

        test_x = emails[test_s: test_e, 0: n_feat]
        test_y = emails[test_s: test_e, -1]

        train_x = np.concatenate([emails[test_e:, 0:n_feat], emails[0:test_s, 0:n_feat]])
        train_y = np.concatenate([emails[test_e:, -1], emails[0:test_s, -1]])

        model.fit(train_x, train_y)
        acc, prec, rec = test_model(model, test_x, test_y)
        print("\t",i+1, "&", np.around(acc, 3), "&", np.around(prec, 3), "&", np.around(rec, 3), "\\\\ \\hline")
        avg_acc += acc
        avg_prec += prec
        avg_rec += rec
    print("\\end{tabular}")
    return avg_acc / 5, avg_prec / 5, avg_rec / 5

