"""
This module contains helper functions for performing forward passes through the
GAN and for computing the gradient prior to backpropagation.

Specifically, this module contains the ReLU activation function, the Leaky ReLU
activation function, the Sigmoid activation function, and their derivatives.
"""
import numpy as np

def relu(node_value_array):
    """
    Perform the ReLU function on a given array. 

    Args:
        node_value_array: A numpy array containing the values of the nodes at a
            given layer, prior to activation.
    Returns:
        The values in the array after passing it through ReLU.
    """
    return np.maximum(0, node_value_array)

def sigmoid(node_value_array):
    """
    Perform the Sigmoid function on a given array. 

    Args:
        node_value_array: A numpy array containing the values of the nodes at a
            given layer, prior to activation.
    Returns:
        The values in the array after passing it through Sigmoid. 
    """
    return 1 / (1 + np.exp(- node_value_array))

def leaky_relu(node_value_array):
    """
    Perform the Leaky RelU function on a given array.

    Args: 
        node_value_array: A numpy array containing the values of the nodes at a
            given layer, prior to activation.
    Returns:
        The values in the array after passing it through Leaky ReLU. 
    """
    alpha = .1
    return (alpha * (node_value_array < 0) + 1 * (node_value_array >= 0)) * node_value_array

def sigmoid_prime(node_value_array):
    """
    Perform the derivative of the Sigmoid function on a given array. This will
    be used for backpropagation.

    Args:
        node_value_array: A numpy array containing the values of the nodes at a
            given layer.
    Returns:
        The values in the array after passing it through Sigmoid'.
    """
    return sigmoid(node_value_array) * (1 - sigmoid(node_value_array))

def relu_prime(node_value_array):
    """
    Perform the derivative of the ReLU function on a given array. This will be
    used for backpropagation.

    Args:
        node_value_array: A numpy array containing the values of the nodes at a
            given layer.
    Returns:
        The values in the array after passing it through ReLU'.    
    """
    return (node_value_array > 0) * 1

def leaky_relu_prime(node_value_array):
    """
    Perform the derivative of the Leaky ReLU function on a given array. This
    will be used for backpropagation.

    Args:
        node_value_array: A numpy array containing the values of the nodes at a
            given layer.
    Returns:
        The values in the array after passing it through (Leaky ReLU)'.
    """
    alpha = .1
    return alpha * (node_value_array < 0) + 1 * (node_value_array >= 0)