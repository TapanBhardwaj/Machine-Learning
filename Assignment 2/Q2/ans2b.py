#################################################################################
#                                                                               #
#                           Code for Question 2(b)                              #
#                                                                               #
#################################################################################

import numpy as np
import math


def sigmoid(Z):
    """
        sigmoid(ndarray) -> ndarray
        
        Applies sigmoid to each entry of the matrix Z
        
        Z: (num_rows, num_cols) input matrix
        
        Returns: Z_hat
            Z_hat: (num_rows, num_cols) Matrix obtained by applying sigmoid to
                   each entry of Z 
    """
    # Initialize the output
    Z_hat = np.zeros(Z.shape)
    
    ########################### YOUR CODE HERE ################################
    
    # Copy your implementation from Ans 2(a) here
    Z_hat = 1.0 / (1.0 + np.exp(-1.0 * Z))
    
    #raise NotImplementedError
    ###########################################################################
    
    return Z_hat
    
    
    
def sigmoid_grad(Z):
    """
        sigmoid_grad(ndarray) -> ndarray
        
        Let Z = sigmoid(X), be matrix obtained by applying sigmoid to another
        matrix X. This function computes sigmoid'(X).
        
        Z: (num_rows, num_cols) Sigmoid output
        
        Returns:
            Z_grad: (num_rows, num_cols) Computed gradient
    """
    # Initialize the output
    Z_grad = np.zeros(Z.shape)
    
    ########################### YOUR CODE HERE ################################
    
    # Copy your implementation from Ans 2(a) here
    Z_grad = np.multiply(Z, 1-Z)
    # ex = math.exp(1)
    # Z_grad = 1 / (2 + ex ** (-Z) + ex ** (Z))
        #(Z) * (1.0 - (Z))
    
    #raise NotImplementedError
    ###########################################################################
    
    return Z_grad   
    


class Linear:
    """
        Class that implements a single linear layer
    """
    
    def __init__(self, num_inputs, num_outputs, act=sigmoid, \
                                                        act_grad=sigmoid_grad):
        """
            __init__(self, int, int, function, function) -> None
            
            num_inputs: Number of features in the input (excluding bias)
            num_outputs: Number of output neurons (excluding bias)
            act: Activation function to use
            act_grad: Function that computes gradient of the activation function
        """ 
        # Initialze variables that will be used later
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.act = act
        self.act_grad = act_grad
        
        ########################### YOUR CODE HERE #############################
    
        # Copy your implementation from Ans 2(a) here
        #self.W = np.ones((num_inputs, num_outputs))
        self.W = np.random.random((self.num_inputs, self.num_outputs))

        #Add bias also
        #self.b = np.ones((num_outputs, 1))
        self.b = np.random.random((self.num_outputs, 1))

        #raise NotImplementedError
        ########################################################################
    
    
    def forward(self, X):
        """
            forward(Linear, ndarray) -> ndarray
            
            Computes the forward pass on this layer
            
            X: (num_examples, num_inputs) Input matrix for this layer. Each row
               corresponds to an example
            
            Returns: out
                out: (num_examples, num_outputs) Computed output activations
        """
        # Some useful variables
        num_examples = X.shape[0]
        
        # Initialze self.out, self is needed beacuse it is used by backward
        self.out = np.zeros((num_examples, self.num_outputs))
        self.input = X # Will be used during backpropagation
        
        ########################### YOUR CODE HERE #############################
    
        # Copy your implementation from ans2a here
        # z = self.act(self.input)
        # z_grad = self.act_grad(z)

        #m = X.dot(self.W)
        m = np.matmul(X, self.W) + self.b.T
        self.out = self.act(m)
        # print(len(m))
        # t = m[0]
        #
        # print(t.shape)
        # print(self.b.shape)
        #return

        # for row in range(len(m)):
        #     t = m[row]
        #     t = t.reshape(self.b.shape)
        #     self.out[row] = sigmoid(t + self.b)[0]

        #self.out = m +

        #raise NotImplementedError
        ########################################################################
        
        return self.out


    def backward(self, delta_out):
        """
            backward(Linear, ndarray) -> ndarray
            Computes the gradient of the weights associated with this layer.
            Returns the error associated with input.

            delta_out: (num_examples, num_output) Error associated with the output units

            Returns: delta_in
                delta_in: (num_examples, num_inputs) Errors associated with the input
                          units
        """
        # Some useful variables
        num_examples = delta_out.shape[0]

        # Initialize the variables
        self.W_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)
        delta_in = np.zeros((num_examples, self.num_inputs))


        ########################### YOUR CODE HERE #############################
    
        # Compute self.W_grad, self.b_grad and delta_in

        # print('deltain',delta_in.shape)
        # print('w',self.W.shape)
        # print('deltaout',delta_out.shape)
        # print('self.input',self.input.shape)
        # print('self.out',self.out.shape)

        intermediate_mat = np.matmul(self.input, self.W)

        m1 = self.act_grad(self.act((intermediate_mat + self.b.T))) * delta_out


        #m1 = delta_out.dot(self.W.T)
        #m1 = delta_out.dot(sigmoid_grad(sigmoid(self.W.T + self.b)))
        #delta_in = np.multiply(m1, sigmoid_grad(sigmoid(self.input[i])))
        delta_in = np.dot(m1, self.W.T)
        #delta_in = np.multiply(m1, self.input[i])

        #a = np.array((self.input[i])).T
        #print('kjn r', a.shape)
        #a = a.reshape(1,self.W.T.shape[1])
        # print(sigmoid(a).shape)
        # print(delta_out.shape)
        # delta_out = delta_out.reshape(1,delta_out.shape[0])

        self.W_grad = np.matmul(self.input.T, m1)
        #b1 = delta_out.T.dot(sigmoid(a))
        # if np.shape(a)[1] == 57:
        #     b = delta_out.dot(sigmoid(a))
        #print('bbefore',b.shape)

        #b = b.reshape(self.W_grad.shape)
        #b1 = b1.T
        #print('bafter',b.shape)
        #self.W_grad += b1#/num_examples
            #np.dot(self.out.T, sigmoid_grad(sigmoid(self.input)))
        # if delta_out.shape[1] == 1:
        #     self.b_grad += delta_out#[0]/num_examples
        # else:
        #     self.b_grad += delta_out[0][0]#/num_examples

        self.b_grad[:, 0] = m1.sum(axis=0)


        #raise NotImplementedError
        ########################################################################
        
        return delta_in


    def step(self, learning_rate=1e-2):
        """
            step(Linear, float) -> None

            Updates the weights of this layer using gradients computed by the
            backward funciton by applying a single step of gradient descent

            learning_rate: The learning rate used for gradient descent
        """
        ########################### YOUR CODE HERE #############################
    
        # Update self.W and self.b based on self.W_grad and self.b_grad

        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad
        #print(self.W_grad)

        #raise NotImplementedError
    
        ########################################################################
        

