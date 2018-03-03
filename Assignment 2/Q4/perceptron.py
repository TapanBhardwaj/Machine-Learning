import copy

import matplotlib.pyplot as plt
import numpy as np

MIS_CLASSIFICATIONS = []


def training(X, Y, iteration=100):
    global MIS_CLASSIFICATIONS
    total_data_points = np.shape(X)[0]

    # initializing weight vector and bias
    W = np.zeros((np.shape(X)[1], 1))
    b = 0
    for i in range(iteration):
        miss = 0

        for j in range(total_data_points):

            if np.sign(np.dot(W.T, X[j].reshape(57, 1))[0][0] + b) == 0 and Y[j] == 1.0:
                # do update
                W = W + (Y[j] * X[j]).reshape(57, 1)
                b = b + Y[j]
                miss = miss + 1
            elif np.sign(np.dot(W.T, X[j].reshape(57, 1))[0][0] + b) != 0 and np.sign(
                            np.dot(W.T, X[j].reshape(57, 1))[0][0] + b) != Y[j]:
                W = W + (Y[j] * X[j]).reshape(57, 1)
                b = b + Y[j]
                miss = miss + 1

        MIS_CLASSIFICATIONS.append(miss)

    return W, b


def get_data_from_file(filepath):
    training_data = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            training_data.append(list(map(float, line.split(", "))))

    training_data = np.asarray(training_data)

    Y = training_data[:, -1:]
    X = training_data[:, :-1]

    return X, Y


def test_accuracy(W, b, X_test, Y_test):
    count = 0
    for j in range(len(X_test)):
        if np.sign(np.dot(W.T, X_test[j].reshape(57, 1))[0][0] + b) == Y_test[j]:
            count += 1

    return 100 * count / len(X_test)


def plot_iter_vs_misclassify():
    plt.plot(MIS_CLASSIFICATIONS, 'b-')

    plt.grid()  # grid is on
    plt.xlabel("Iteration No.")
    plt.ylabel("Total mis-classifications")
    plt.show()


def highest_five(W):
    max_list = []
    for i in range(5):
        max_list.append(np.argmax(W))
        W[np.argmax(W)] = -np.inf

    return max_list


def lowest_five(W):
    min_list = []
    for i in range(5):
        min_list.append(np.argmin(W))
        W[np.argmin(W)] = np.inf

    return min_list


if __name__ == '__main__':
    filepath_train = '../Q1/data/SpambaseFull/train.txt'
    filepath_test = '../Q1/data/SpambaseFull/test.txt'

    X_train, Y_train = get_data_from_file(filepath_train)
    X_test, Y_test = get_data_from_file(filepath_test)

    weight, bias = training(X_train, Y_train, 100)
    plot_iter_vs_misclassify()

    print("Train accuracy is : {}%".format(round(test_accuracy(weight, bias, X_train, Y_train)), 3))
    print("Test accuracy is : {}%".format(round(test_accuracy(weight, bias, X_test, Y_test)), 3))

    print("Indices having highest 5 weight : {}".format(highest_five(copy.deepcopy(weight))))
    print("Indices having lowest 5 weight : {}".format(lowest_five(copy.deepcopy(weight))))
