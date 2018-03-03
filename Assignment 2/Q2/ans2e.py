#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


class One_hidden_Layer_Network(torch.nn.Module):
    def __init__(self, Data_input, Hidden_layer_input, Data_out):
        super(One_hidden_Layer_Network, self).__init__()
        self.linear_layer1 = torch.nn.Linear(Data_input, Hidden_layer_input)
        self.linear_layer2 = torch.nn.Linear(Hidden_layer_input, Data_out)

    def forward(self, X):
        preact1 = self.linear_layer1(X)
        X2 = preact1.sigmoid()
        preact2 = self.linear_layer2(X2)
        Y = preact2.sigmoid()
        return Y


if __name__ == '__main__':
    print()
    filepath_train = '../Q1/data/SpambaseFull/train.txt'
    training_data = []
    with open(filepath_train) as fp:
        for cnt, line in enumerate(fp):
            training_data.append(list(map(float, line.split(", "))))

    training_data = np.asarray(training_data)

    Y_train = training_data[:, -1:]
    X_train = training_data[:, :-1]

    for i in range(len(Y_train)):
        if (Y_train[i] == -1):
            Y_train[i] = 0

    filepath_test = '../Q1/data/SpambaseFull/test.txt'
    testing_data = []
    with open(filepath_test) as fp:
        for cnt, line in enumerate(fp):
            testing_data.append(list(map(float, line.split(", "))))

    testing_data = np.asarray(testing_data)

    Y_test = testing_data[:, -1:]
    X_test = testing_data[:, :-1]

    for i in range(len(Y_test)):
        if (Y_test[i] == -1):
            Y_test[i] = 0

    N, Data_input, Hidden_layer_input, Data_output = X_train.shape[0], X_train.shape[1], 20, 1

    X_train_variable = Variable(torch.from_numpy(X_train), requires_grad=False)
    X_test_variable = Variable(torch.from_numpy(X_test), requires_grad=False)
    Y_train_variable = Variable(torch.from_numpy(Y_train))
    Y_test_variable = Variable(torch.from_numpy(Y_test))
    X_train_variable = X_train_variable.float()
    Y_train_variable = Y_train_variable.float()
    X_test_variable = X_test_variable.float()
    Y_test_variable = Y_test_variable.float()

    training_error = []
    testing_error = []
    lambda_val = [0.0001, 0.001, 0.01, 0.1, 1]

    model = One_hidden_Layer_Network(Data_input, Hidden_layer_input, Data_output)
    criteria = torch.nn.MSELoss()

    for l in lambda_val:

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=l)

        for i in range(1000):
            Y_pred = model.forward(X_train_variable)

            loss = criteria(Y_pred, Y_train_variable)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Y_pred_train = model.forward(X_train_variable)

        Y_pred_test = model.forward(X_test_variable)
        z1 = Y_pred_train.data.numpy()
        z2 = Y_pred_test.data.numpy()
        l1 = np.ones(Y_pred_train.shape)
        l2 = np.ones(Y_pred_test.shape)
        for i1 in range(len(Y_pred_train)):
            if (z1[i1] >= 0.5):
                l1[i1] = 1
            else:
                l1[i1] = 0

        for i2 in range(len(Y_pred_test)):
            if (z2[i2] >= 0.5):
                l2[i2] = 1
            else:
                l2[i2] = 0
        match_train = 0
        match_test = 0
        for i3 in range(len(Y_pred_train)):
            if (l1[i3] != Y_train[i3]):
                match_train += 1
        for i4 in range(len(Y_pred_test)):
            if (l2[i4] != Y_test[i4]):
                match_test += 1

        training_error.append((match_train / len(Y_pred_train)))
        testing_error.append((match_test / len(Y_pred_test)))

    lambda_log = [-4, -3, -2, -1, 0]

    training_accuracy = []
    test_accuracy = []

    for k in range(len(training_error)):
        training_accuracy.append(100 * (1.0 - training_error[k]))

    for k in range(len(testing_error)):
        test_accuracy.append(100 * (1.0 - testing_error[k]))

    plt.plot(lambda_log, training_accuracy, label='Training_Accuracy')
    plt.plot(lambda_log, test_accuracy, label='Testing_Accuracy')

    plt.legend()
    plt.xlabel('Values of log of lamba')
    plt.ylabel('Accuracy Percentage')

    plt.show()
