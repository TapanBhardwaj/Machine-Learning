#!/usr/bin/env python

import ans2c as a2c
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

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
    net = a2c.Net2c()
    epoch = []
    cost_at_epoch = []
    min_cost = 0

    for i in range(100000):

        preds = net.forward(X_train)
        for j in range(len(preds)):
            if (preds[j] > 0.5):
                preds[j] = 1
            else:
                preds[j] = 0

        cost = net.backward(preds, Y_train)
        epoch.append(i)
        cost_at_epoch.append(cost)
        if cost[0] <= 2.5:
            break

        net.step()
    print(min_cost)

    Y_pred_train = net.forward(X_train)
    for j in range(len(Y_pred_train)):
        if (Y_pred_train[j] > 0.5):
            Y_pred_train[j] = 1
        else:
            Y_pred_train[j] = 0

    match_count = 0
    for i in range(len(Y_pred_train)):
        if Y_pred_train[i] == Y_train[i]:
            match_count += 1
    print('Training accuracy:', (match_count * 100) / len(Y_train))

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

    Y_pred_test = net.forward(X_test)
    for j in range(len(Y_pred_test)):
        if (Y_pred_test[j] > 0.5):
            Y_pred_test[j] = 1
        else:
            Y_pred_test[j] = 0

    match_count = 0
    for i in range(len(Y_pred_test)):
        if Y_pred_test[i] == Y_test[i]:
            match_count += 1
    print('Testing accuracy:', (match_count * 100) / len(Y_test))

    plt.plot(epoch, cost_at_epoch)
    plt.xlabel('Iteration number')
    plt.ylabel('Cost at each iteration')
    plt.show()
