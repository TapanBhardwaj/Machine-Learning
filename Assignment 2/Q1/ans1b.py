################################################################################
#                                                                              #
#                           Code for question 1(b)                             #
#                                                                              #
################################################################################

import ans1a as a1a
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split

# stopping solvers.qp progress
solvers.options["show_progress"] = False


def linear_kernel(x1, x2):
    """
        linear_kernel(ndarray, ndarray) -> float
        
        Computes the linear kernel of the specified examples.
        
        x1: (1, num_features) Input vector
        x2: (1, num_features) Input vector
        
        Returns: value
            value: The kernel value computed by applying the linear kernel
    """
    return np.matmul(x1, x2.T)


def svm_train(X, Y, kernel=linear_kernel):
    """
        svm_train(ndarray, ndarray, function) -> ndarray, float
        
        Trains the hard SVM on provided data.
        
        X: (num_examples, num_features) Input feature matrix
        Y: (num_examples, 1) Labels
        kernel: Function that computes kernel(x1, x2)
        
        Returns: (alphas, b)
            alphas: (num_examples, 1) alphas obtained by solving SVM
            b: bias term for SVM
    """
    # Some useful variables
    n, d = X.shape

    ############################## YOUR CODE HERE ##############################

    # Prepare your optimization problem here. 
    # Install cvxopt by using:
    #   pip install cvxopt
    # Use cvxopt (http://cvxopt.org/examples/tutorial/qp.html) to solve it.
    # Use the solution to obtain alphas and b

    P = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            P[i][j] = Y[i] * Y[j] * kernel(X[i], X[j])

    P = matrix(P)
    q = matrix(-np.ones((n, 1)))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(Y.reshape(1, -1))
    b = matrix(np.zeros(1))

    solution = solvers.qp(P, q, G, h, A, b)  # using same name of variables as given in http://cvxopt.org/examples/
    alphas = np.array(solution['x'])

    # bias term evaluation

    # finding min when y[i] = 1
    min_list = []
    max_list = []

    for i in range(n):
        SUM = 0
        for j in range(n):
            SUM += alphas[j] * Y[j] * kernel(X[j], X[i])
        if Y[i] == 1:
            min_list.append(SUM)
        else:
            max_list.append(SUM)

    bias = - 0.5 * (min(min_list) + max(max_list))

    return alphas, bias


def svm_predict(X_train, Y_train, X, alphas, b, kernel=linear_kernel):
    """
        svm_predict(ndarray, ndarray, ndarray, ndarray, float, function) -> \
                                                         float, float
        
        Predicts the output labels Y based on input X and trained SVM
        
        X_train: (num_examples, num_features) Training feature matrix
        Y_train: (num_examples, 1) Training labels
        X: (1, num_features) Features of input test example
        alphas: (num_examples, 1) alphas obtained by training SVM
        b: Bias term
        kernel: Kernel function to use (same as svm_train)
        
        Returns: (Y, fval)
            Y: Predicted label {+1, -1} for X
            fval: The value of function which was thresholded to get Y
    """
    # Some useful variables
    n, d = X_train.shape

    # Intialize output
    fval = 0  # The value equivalent to wx + b
    Y = 0  # Label obtained by thresholding fval

    # ############################# YOUR CODE HERE ##############################

    # Compute the output prediction Y
    fval += b
    for i in range(n):
        fval += alphas[i] * Y_train[i] * kernel(X_train[i], X)

    if fval <= -1:
        Y = -1
    if fval >= 1:
        Y = 1
    ############################################################################

    return Y, fval


def show_decision_boundary(X, Y, X_train, Y_train, alphas, b, \
                           kernel=linear_kernel):
    """
        show_decision_boundary(ndarray, ndarray, ndarray, ndarray, ndarray, \
                                        float, function) -> None
    
        Shows decision boundary by plotting regions of positive and negative
        classes
        
        X: (num_examples, num_features) Feature matrix
        Y: (num_examples, 1) Label matrix
        X_train: (num_train_examples, 1) Feature matrix used for training
        Y_train: (num_train_examples, 1) Labels from training set
        alphas: (num_train_examples, 1) alphas obtained by training SVM
        b: Bias term
        kernel: Kernel function to use (same as svm_train)
    """
    # Some useful variables
    n, d = X.shape

    # Obtain the minimum and maximum coordinates
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    # Plot the prediction map
    colors = ['rs', 'bs']
    for x in np.arange(x_min, x_max, 0.2).tolist():
        for y in np.arange(y_min, y_max, 0.2).tolist():
            label, _ = svm_predict(X_train, Y_train, np.asarray([[x, y]]), \
                                   alphas, b, kernel)
            plt.plot(x, y, colors[int((label + 1) / 2)])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    # Use ans1a to generate synthetic data as mentioned in question 1(b)
    # Now use data_gen to generate synthetic data as specified in question 1(a)
    X, Y = a1a.data_gen(n=1000)

    # Split into train_data [80%] and test_data [20%]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Use svm_train to train on train_data using linear_kernel
    alphas, b = svm_train(X_train, Y_train)

    # Use svm_predict to obtain accuracy on train_data
    counter = 0
    for i in range(len(X_train)):
        label, _ = svm_predict(X_train, Y_train, X_train[i], alphas, b)
        if label == Y_train[i]:
            counter += 1

    accuracy = 100 * counter / len(X_train)
    print("Percentage accuracy of train data is {}%".format(accuracy))

    # Use svm_predict to obtain accuracy on test_data

    counter = 0
    for i in range(len(X_test)):
        label, _ = svm_predict(X_train, Y_train, X_test[i], alphas, b)
        if label == Y_test[i]:
            counter += 1

    accuracy = 100 * counter / len(X_test)
    print("Percentage accuracy of test data is {}%".format(accuracy))

    # Show the decision region using show_decision_boundary
    show_decision_boundary(X, Y, X_train, Y_train, alphas, b)
