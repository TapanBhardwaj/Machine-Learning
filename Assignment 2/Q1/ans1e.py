################################################################################
#                                                                              #
#                           Code for question 1(e)                             #
#                                                                              #
################################################################################

import ans1a as a1a
import ans1b as a1b
import ans1d as a1d
import numpy as np
from sklearn.model_selection import train_test_split


def rbf_kernel(x1, x2):
    """
        rbf_kernel(ndarray, ndarray) -> float
        
        Computes the RBF kernel of the specified examples.
        
        x1: (1, num_features) Input vector
        x2: (1, num_features) Input vector
        
        Returns: value
            value: The kernel value computed by applying the RBF kernel
    """
    # Initialize the value
    val = 0.0

    ############################## YOUR CODE HERE ##############################

    # Compute the kernel value

    val = -0.1 * ((np.linalg.norm(x1 - x2)) ** 2)
    val = np.exp(val)

    ############################################################################

    return val


if __name__ == '__main__':
    # Generate 1000 synthetic data points as in 1(d)
    X, Y = a1a.data_gen(n=1000, f=a1d.complex_fn)

    # Implement rbf_kernel function

    # Split into train_data [80%] and test_data [20%]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Use svm_train to train on train_data using rbf_kernel
    alphas, b = a1b.svm_train(X_train, Y_train, kernel=rbf_kernel)

    # Use svm_predict to obtain accuracy on train_data
    count = 0
    for i in range(len(X_train)):
        label, _ = a1b.svm_predict(X_train, Y_train, X_train[i], alphas, b, kernel=rbf_kernel)
        if label == Y_train[i]:
            count = count + 1

    accuracy = 100 * count / len(X_train)
    print("Percentage accuracy of train data is {}%".format(accuracy))

    # Use svm_predict to obtain accuracy on test_data

    count = 0
    for i in range(len(X_test)):
        label, _ = a1b.svm_predict(X_train, Y_train, X_test[i], alphas, b, kernel=rbf_kernel)
        if label == Y_test[i]:
            count = count + 1

    accuracy = 100 * count / len(X_test)
    print("Percentage accuracy of test data is {}%".format(accuracy))

    # Show the decision region using show_decision_boundary
    a1b.show_decision_boundary(X, Y, X_train, Y_train, alphas, b, kernel=rbf_kernel)
