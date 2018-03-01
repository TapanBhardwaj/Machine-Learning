import matplotlib.pyplot as plt
import numpy as np

MIS_CLASSIFICATIONS = []


def training(X, Y, iteration=100):
    global MIS_CLASSIFICATIONS
    total_data_points = np.shape(X)[0]

    # initializing weight vector and bias
    W = np.zeros((np.shape(X)[1], 1))
    print("adaasa", np.matmul(W.T, X[1].reshape(57, 1)))
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


def plot_iter_vs_misclassify():
    # blue dot, green dot
    plt.plot(MIS_CLASSIFICATIONS, "ro")

    plt.grid()  # grid is on
    plt.xlabel("Iteration No.")
    plt.ylabel("Total mis-classifications")
    plt.show()


if __name__ == '__main__':
    filepath = '../Q1/data/SpambaseFull/train.txt'
    training_data = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            training_data.append(list(map(float, line.split(", "))))

    training_data = np.asarray(training_data)

    Y = training_data[:, -1:]
    X = training_data[:, :-1]
    print(type(X[0]), np.shape(X[0]))
    print(X[1].reshape(57, 1), np.shape(X[1].reshape(57, 1)))

    training(X, Y, 100)

    print(MIS_CLASSIFICATIONS)
    plot_iter_vs_misclassify()

    # print(MIS_CLASSIFICATIONS)

    # print(np.shape(Y), Y)
    # print(np.shape(X), X)
