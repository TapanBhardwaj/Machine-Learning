import ans1c as a1c
import ans1e as a1e
import numpy as np




def get_data_from_file(filepath):
    # filepath = '../Q1/data/SpambaseFull/train.txt'
    training_data = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            training_data.append(list(map(float, line.split())))

    training_data = np.asarray(training_data)

    Y = training_data[:, -1:]
    X = training_data[:, :-1]

    return X, Y


def accuracy(X_train, Y_train, X_test, Y_test, c):
    alphas, bias = a1c.svm_train(X_train, Y_train, c, kernel=a1e.rbf_kernel)

    counter = 0
    for i in range(len(X_test)):
        label, _ = a1c.svm_predict(X_train, Y_train, X_test[i], alphas, bias, kernel=a1e.rbf_kernel)
        if label == Y_test[i]:
            counter += 1

    accuracy = 100 * counter / len(X_test)
    return accuracy


if __name__ == '__main__':
    C = [0.01, 0.1, 1, 10, 100]

    for c in C:

        accuracy_list = []
        for i in range(5):
            path_train = "../Q1/data/SpambaseFolds/Fold" + str(i + 1) + "/cv-train.txt"
            path_test = "../Q1/data/SpambaseFolds/Fold" + str(i + 1) + "/cv-test.txt"

            X_train, Y_train = get_data_from_file(path_train)
            X_test, Y_test = get_data_from_file(path_test)

            accuracy_percent = accuracy(X_train, Y_train, X_test, Y_test, c)
            accuracy_list.append(accuracy_percent)

        print("Percentage accuracy for c = {} is {}%".format(c, sum(accuracy_list)/5))
