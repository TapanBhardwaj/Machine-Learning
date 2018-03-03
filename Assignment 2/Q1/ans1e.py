################################################################################
#                                                                              #
#                           Code for question 1(e)                             #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import ans1a as a1a
import ans1d as a1d
import ans1b as a1b


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

    # Use a1b.svm_train with rbf_kernel on 80% of data - train_data
    # Compute accuracy on train_data and test_data

    # Show the decision region using a1b.show_decision_boundary
