################################################################################
#                                                                              #
#                           Code for question 1(d)                             #
#                                                                              #
################################################################################

import ans1a as a1a
from numpy import linalg as la


def complex_fn(x):
    """
        complex_fn(ndarray) -> {0, 1}

        Complex function that assigns class to x as explained in question 1(d)

        Returns: y
            y: Class label {+1, -1}
    """
    if la.norm(x) < 3:
        return +1
    return -1


if __name__ == '__main__':
    # Use data_gen to generate synthetic data as specified in question 1(a)
    X, Y = a1a.data_gen(n=1000, f=complex_fn)

    # Plot the generated data using matplotlib
    a1a.plot_data_points(X, Y)
