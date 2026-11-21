import numpy as np


def relu(z):
    """Rectified Linear Unit activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative of ReLU with respect to pre-activation z."""
    return (z > 0.0).astype(np.float64)


def softmax(z):
    """Numerically stable row-wise Softmax activation."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
