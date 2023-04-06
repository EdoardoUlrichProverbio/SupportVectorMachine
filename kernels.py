import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def poly_kernel(x1, x2, degree=3, coef = 1.0):
    return (coef + np.dot(x1, x2.T)) ** degree

def rbf_kernel(x1, x2, gamma):
    dist_matrix = np.sum(x1**2, axis=1)[:, np.newaxis] + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return np.exp(-gamma * dist_matrix)

def sigmoid_kernel(x1, x2, theta, coef):
    return np.tanh(np.dot(x1, x2.T) * theta + coef)