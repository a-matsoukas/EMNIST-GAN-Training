"""

"""
import numpy as np

class BasicDiscriminator:
    """
    Randomly initialized (untrained) discriminator of fixed size.

    Attributes:
        _Theta1: A numpy array of weights between the input layer and the first
            hidden layer.
        _Theta2: A numpy array of weights between the first hidden layer and the
            output layer.
    """
    
    def __init__(self):
        """
        Randomly initialize the weights of a basic discriminator multi-layer
        perceptron (MPL) of fixed size. Including the bias terms, this MLP has
        three layers with 785 nodes, 5 nodes, and 1 node.

        Args:
            None
        Returns:
            No Return Value
        """
        self._Theta1 = 10 * np.random.rand(785, 4) - 5
        self._Theta2 = 10 * np.random.rand(5, 1) - 5

    @property
    def Theta1(self):
        """
        Return the matrix of weights, Theta1, which is a private attribute.

        Args:
            None
        Returns:
            self._Theta1: a 785 x 4 numpy array of weights
        """
        return self._Theta1

    @property
    def Theta2(self):
        """
        Return the matrix of weights, Theta2, which is a private attribute.

        Args:
            None
        Returns:
            self._Theta2 = a 5 x 1 numpy array of weights
        """
        return self._Theta2


class BasicGenerator:
    """
    Randomly initialized (untrained) generator of fixed size.

    Attributes:
        _Phi1: A numpy array of weights between the input layer and the first
            hidden layer.
        _Phi2: A numpy array of weights between the first hidden layer and the
            second hidden layer.
        _Phi3: A numpy array of weights between the second hidden layer and the
            final layer.
    """

    def __init__(self):
        """
        Randomly initialize the weights of a basic generator multi-layer
        perceptron (MPL) of fixed size. Including the bias terms, this MLP has
        four layers with 5 nodes, 50 nodes, 250 nodes, and 748 nodes.

        Args:
            None
        Returns:
            No Return Value
        """
        self._Phi1 = 10 * np.random.rand(5, 49) - 5
        self._Phi2 = 10 * np.random.rand(50, 249) - 5
        self._Phi3 = 10 * np.random.rand(250, 784) - 5
    
    @property
    def Phi1(self):
        """
        Return the matrix of weights, Phi1, which is a private attribute.

        Args:
            None
        Returns:
            self._Phi1 = a 5 x 49 numpy array of weights
        """
        return self._Phi1

    @property
    def Phi2(self):
        """
        Return the matrix of weights, Phi2, which is a private attribute.

        Args:
            None
        Returns:
            self._Phi2 = a 50 x 249 numpy array of weights
        """
        return self._Phi2

    @property
    def Phi3(self):
        """
        Return the matrix of weights, Phi3, which is a private attribute.

        Args:
            None
        Returns:
            self._Phi3 = a 250 x 784 numpy array of weights
        """
        return self._Phi3

