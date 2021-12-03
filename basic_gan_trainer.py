"""

"""
import numpy as np
from activation_functions import relu, sigmoid, relu_prime, sigmoid_prime

class BasicGANTrainer:
    """
    
    """

    def __init__(self):
        """
        
        """
        self._lambda = .1

        self._Din_bias = None
        self._D1_bias = None
        self._Dout = None
        self._Theta1 = np.random.rand(4, 785) - .5
        self._Theta2 = np.random.rand(1, 5) - .5

        self._Gin_bias = None
        self._G1_bias = None
        self._G2_bias = None
        self._Gout = None
        self._Phi1 = np.random.rand(49, 5) - .5
        self._Phi2 = np.random.rand(249, 50) - .5
        self._Phi3 = np.random.rand(784, 250) - .5

    @property
    def Din_bias(self):
        """
        
        """
        return self._Din_bias

    @property
    def D1_bias(self):
        """
        
        """
        return self._D1_bias

    @property
    def Dout(self):
        """
        
        """
        return self._Dout

    @property
    def Gin_bias(self):
        """
        
        """
        return self._Gin_bias

    @property
    def G1_bias(self):
        """
        
        """
        return self._G1_bias

    @property
    def G2_bias(self):
        """
        
        """
        return self._G2_bias

    @property
    def Gout(self):
        """
        
        """
        return self._Gout

    @property
    def Theta1(self):
        """
        
        """
        return self._Theta1

    @property
    def Theta2(self):
        """
        
        """
        return self._Theta2

    @property
    def Phi1(self):
        """
        
        """
        return self._Phi1

    @property
    def Phi2(self):
        """
        
        """
        return self._Phi2

    @property
    def Phi3(self):
        """
        
        """
        return self._Phi3

    def disc_forward_pass(self, train_ex):
        """
        
        """
        self._Din_bias = np.insert(train_ex, 0, 1, axis=0)

        D1 = relu(np.matmul(self.Theta1, self.Din_bias))
        self._D1_bias = np.insert(D1, 0, 1, axis=0)

        self._Dout = sigmoid(np.matmul(self.Theta2, self.D1_bias))

        return self.Dout

    def gen_forward_pass(self, train_ex):
        """
        
        """
        self._Gin_bias = np.insert(train_ex, 0, 1, axis=0)

        G1 = relu(np.matmul(self.Phi1, self.Gin_bias))
        self._G1_bias = np.insert(G1, 0, 1, axis=0)

        G2 = relu(np.matmul(self.Phi2, self.G1_bias))
        self._G2_bias = np.insert(G2, 0, 1, axis=0)

        self._Gout = relu(np.matmul(self.Phi3, self.G2_bias))

        return self.Gout

    def calc_disc_grad():
        """
        
        """
        
    def calc_gen_grad():
        """
        
        """

    def update_Thetas():
        """
        
        """
    
    def update_Phis():
        """
        
        """


