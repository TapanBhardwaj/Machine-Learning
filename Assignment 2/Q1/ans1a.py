################################################################################
#                                                                              #
#                           Code for question 1(a)                             #
#                                                                              #
################################################################################

import matplotlib.pyplot as plt
import numpy as np


def dummy_fn(x):
    """
        dummy_fn(ndarray) -> {0, 1}

        Simple function that assigns class to x as: y = sign(sum(x))

        Returns: y
            y: Class label {+1, -1}
    """
    if np.sum(x) == 0:
        return -1
    return np.sign(np.sum(x))


def data_gen(n=100, f=dummy_fn, lims=[5, 5]):
    """
        data_gen(int, function, list) -> (ndarray, ndarray)

        Generates synthetic data by using the classification rule specified by
        the function f.

        n: Number of data points to generate
        f: Function that takes a d-dimensional vector x as input and produces a
           the class label for that point {+1, -1}
        lims: lims[i] is used to get upper and lower limit on domain for the ith
              dimension. The ith dimension of x will be uniformly sampled from
              the interval [-lim[i], lim[i]] for all examples
        Length of dims (=d) should be taken as the dimension of input vectors.

        Returns: (X, Y)
            X: (n, d) Feature matrix
            Y: (n, 1) Label matrix
    """
    # Some useful variables
    d = len(lims)

    # Intialize the data
    X = np.zeros((n, d))
    Y = np.zeros((n, 1))

    for i in range(n):

        for dim in range(len(lims)):
            X[i][dim] = np.random.uniform(-lims[dim], lims[dim])

        Y[i] = f(X[i])

    return (X, Y)


def plot_data_points(X, Y):
    """
            plot_data_points(ndarray, ndarray)

            Simple function that plots datapoints with different colors according to labels
    """
    # Plot the generated data using matplotlib

    list_label_positive_x = []
    list_label_positive_y = []
    list_label_negative_x = []
    list_label_negative_y = []

    for i in range(1000):
        if Y[i] == 1:
            list_label_positive_x.append(X[i][0])
            list_label_positive_y.append(X[i][1])
        else:
            list_label_negative_x.append(X[i][0])
            list_label_negative_y.append(X[i][1])

    # blue dot, green dot
    plt.plot(list_label_positive_x, list_label_positive_y, "ro", label='label 1')
    plt.plot(list_label_negative_x, list_label_negative_y, "g^", label='label -1')

    plt.grid()  # grid is on
    plt.legend()  # add legend based on line labels. see left top corner
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    # Implement the data_gen function first

    # Now use data_gen to generate synthetic data as specified in question 1(a)
    X, Y = data_gen(n=1000)

    # Plot the generated data using matplotlib
    plot_data_points(X, Y)
