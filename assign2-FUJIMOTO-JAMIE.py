#!/usr/bin/env python2

from __future__ import division
import numpy as np
import random
import sys

__author__ = 'Jamie Fujimoto'


def get_data(filename):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def kernel_func(kernel, Xi, Xj):
    Kij = np.dot(Xi, Xj)
    if kernel == 'linear':
        return Kij
    elif kernel == 'quadratic':
        return Kij**2


def compute_ay(a, y):
    """
    Used to vectorize sum([a[i] * y[i] for i in xrange(n)])

    Returns a vector using element-wise multiplication between vector a and y
    """
    ay = np.multiply(a, y)
    return ay


def compute_error(kernel, X, y, a, n, k):
    """
    Computes the error between the predicted value and the true value for the point X[k, :]

    Returns a scalar
    """
    ay = compute_ay(a, y)
    Kjk = kernel_func(kernel, X, X[k, :])
    return np.dot(ay, Kjk) - y[k]


def compute_bi(kernel, X, y, a, i):
    ay = compute_ay(a, y)
    Kji = kernel_func(kernel, X, X[i, :])
    return y[i] - np.dot(ay, Kji)  # using Eq. (21.33)


def compute_weight(X, y, a, i):
    return a[i] * y[i] * X[i]


def predict(a, indexes, kernel, X, y, b, z):
    y_hat = sum([a[i] * y[i] * kernel_func(kernel, X[i], z) for i in indexes]) + b
    return 1 if y_hat > 0 else -1


def SMO(X, y, C, kernel, eps):
    n = X.shape[0]
    a = np.zeros(n)
    count = 0

    while True:
        count += 1
        a_prev = a.copy()
        for j in xrange(n):
            # i = random index in range 1,...,n such that i != j
            choices = range(0, j) + range(j+1, n)
            i = random.choice(choices)

            # Compute Kij based on kernel type (linear or quadratic)
            Kij = kernel_func(kernel, X[i, :], X[i, :]) + kernel_func(kernel, X[j, :], X[j, :]) - \
                  (2 * kernel_func(kernel, X[i, :], X[j, :]))

            if Kij == 0:
                continue

            aj_prime, ai_prime = a[j].copy(), a[i].copy()

            # Compute L and H based on the two cases
            # case 1: yi != yj
            if y[i] != y[j]:
                L = max(0.0, aj_prime - ai_prime)
                H = min(C, C - ai_prime + aj_prime)

            # case 2: yi == yj
            elif y[i] == y[j]:
                L = max(0.0, ai_prime + aj_prime - C)
                H = min(C, ai_prime + aj_prime)

            # Compute Ei and Ej
            Ei = compute_error(kernel, X, y, a, n, i)
            Ej = compute_error(kernel, X, y, a, n, j)

            a[j] = aj_prime + ((y[j] * (Ei - Ej)) / Kij)

            if a[j] < L:
                a[j] = L
            elif a[j] > H:
                a[j] = H

            a[i] = ai_prime + (y[i] * y[j] * (aj_prime - a[j]))
        if np.linalg.norm(a - a_prev) <= eps:
            break
        if count % 100 == 0:
            print np.linalg.norm(a - a_prev)
    return a


if __name__ == "__main__":
    filename = sys.argv[1]
    C = float(sys.argv[2])
    kernel = sys.argv[3]
    eps = float(sys.argv[4])

    X, y = get_data(filename)
    a = SMO(X, y, C, kernel, eps)
    n = a.shape[0]

    # Get support vector indexes
    indexes = [i for i in xrange(n) if a[i] > 0]

    ai = np.array([a[i] for i in indexes])  # Create ai vector
    print "ai.shape: {0}".format(ai.shape)
    yi = np.array([y[i] for i in indexes])  # Create yi vector
    print "yi.shape: {0}".format(yi.shape)
    Xi = np.array([X[i, :] for i in indexes])  # Create Xi matrix
    print "Xi.shape: {0}".format(Xi.shape)

    aiyi = compute_ay(ai, yi)
    print "aiyi.shape: {0}".format(aiyi.shape)

    # # Compute bias
    # bi_vec = [compute_bi(kernel, X, y, a, i) for i in indexes]
    # bias = sum(bi_vec) / len(bi_vec)

    # # Compute w if linear kernel
    # if kernel == "linear":
    #     weights = [compute_weight(X, y, a, i) for i in indexes]
    #     d = X.shape[1]
    #     weight = np.zeros(d)
    #     for w in weights:
    #         for i in xrange(d):
    #             weight[i] += w[i]
    #
    #
    # # Compute accuracy
    # pred = [predict(a, indexes, kernel, X, y, bias, X[i]) for i in xrange(n)]
    # count = 0
    # for i in xrange(n):
    #     if pred[i] == y[i]:
    #         count += 1
    # accuracy = count / n
    #
    # with open("assign2-FUJIMOTO-JAMIE.txt", "w") as f:
    #     # Print support vector indexes
    #     print "The support vectors are:"
    #     f.write("The support vectors are:\n")
    #     for i in indexes:
    #         print i, a[i]
    #         f.write("{0} {1}\n".format(i, a[i]))
    #     print "Number of support vectors: {0}\n".format(len(indexes))
    #     f.write("Number of support vectors: {0}\n".format(len(indexes)))
    #
    #     # Print bias
    #     print "Bias:", bias
    #     f.write("Bias: {0}\n".format(bias))
    #
    #     # Print w for linear kernel
    #     if kernel == "linear":
    #         print "w:", weight
    #         f.write("w: {0}\n".format(weight))
    #
    #     # Print accuracy
    #     print "Accuracy:", accuracy
    #     f.write("Accuracy: {0}\n".format(accuracy))