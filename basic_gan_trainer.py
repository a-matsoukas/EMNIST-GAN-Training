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
        self._lambda_Theta4 = .00005
        self._lambda_Theta3 = .0005
        self._lambda_Theta2 = .005
        self._lambda_Theta1 = .05

        self._lambda_Phi4 = .0001
        self._lambda_Phi3 = .001
        self._lambda_Phi2 = .01
        self._lambda_Phi1 = .1

        self._Din_bias = None
        self._D1_bias = None
        self._D2_bias = None
        self._D3_bias = None
        self._Dout = None
        self._Theta1 = (np.random.rand(499, 785) - .5) / 25
        self._Theta2 = (np.random.rand(249, 500) - .5) / 25
        self._Theta3 = (np.random.rand(49, 250) - .5) / 25
        self._Theta4 = (np.random.rand(1, 50) - .5) / 25
        self._Delta1 = np.zeros(self.Theta1.shape)
        self._Delta2 = np.zeros(self.Theta2.shape)
        self._Delta3 = np.zeros(self.Theta3.shape)
        self._Delta4 = np.zeros(self.Theta4.shape)

        self._Gin_bias = None
        self._G1_bias = None
        self._G2_bias = None
        self._G3_bias = None
        self._Gout = None
        self._Phi1 = (np.random.rand(49, 50) - .5) / 25
        self._Phi2 = (np.random.rand(249, 50) - .5) / 25
        self._Phi3 = (np.random.rand(249, 250) - .5) / 25
        self._Phi4 = (np.random.rand(784, 250) - .5) / 25
        self._Gamma1 = np.zeros(self.Phi1.shape)
        self._Gamma2 = np.zeros(self.Phi2.shape)
        self._Gamma3 = np.zeros(self.Phi3.shape)
        self._Gamma4 = np.zeros(self.Phi4.shape)

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
    def D2_bias(self):
        """
        
        """
        return self._D2_bias

    @property
    def D3_bias(self):
        """
        
        """
        return self._D3_bias

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
    def G3_bias(self):
        """
        
        """
        return self._G3_bias

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
    def Theta3(self):
        """
        
        """
        return self._Theta3

    @property
    def Theta4(self):
        """
        
        """
        return self._Theta4

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

    @property
    def Phi4(self):
        """
        
        """
        return self._Phi4

    @property
    def Delta1(self):
        """
        
        """
        return self._Delta1

    @property
    def Delta2(self):
        """
        
        """
        return self._Delta2

    @property
    def Delta3(self):
        """
        
        """
        return self._Delta3

    @property
    def Delta4(self):
        """
        
        """
        return self._Delta4

    @property
    def Gamma1(self):
        """
        
        """
        return self._Gamma1

    @property
    def Gamma2(self):
        """
        
        """
        return self._Gamma2

    @property
    def Gamma3(self):
        """
        
        """
        return self._Gamma3

    @property
    def Gamma4(self):
        """
        
        """
        return self._Gamma4
    
    def disc_forward_pass(self, train_ex):
        """
        
        """
        self._Din_bias = np.insert(train_ex, 0, 1, axis=0)

        D1 = relu(np.matmul(self.Theta1, self.Din_bias))
        self._D1_bias = np.insert(D1, 0, 1, axis=0)

        D2 = relu(np.matmul(self.Theta2, self.D1_bias))
        self._D2_bias = np.insert(D2, 0, 1, axis=0)

        D3 = relu(np.matmul(self.Theta3, self.D2_bias))
        self._D3_bias = np.insert(D3, 0, 1, axis=0)

        Dout_temp = sigmoid(np.matmul(self.Theta4, self.D3_bias))
        self._Dout = np.maximum(np.minimum(Dout_temp, .9999), .0001)

        return self.Dout

    def gen_forward_pass(self, train_ex):
        """
        
        """
        self._Gin_bias = np.insert(train_ex, 0, 1, axis=0)

        G1 = relu(np.matmul(self.Phi1, self.Gin_bias))
        self._G1_bias = np.insert(G1, 0, 1, axis=0)

        G2 = relu(np.matmul(self.Phi2, self.G1_bias))
        self._G2_bias = np.insert(G2, 0, 1, axis=0)

        G3 = relu(np.matmul(self.Phi3, self.G2_bias))
        self._G3_bias = np.insert(G3, 0, 1, axis=0)

        self._Gout = (sigmoid(np.matmul(self.Phi4, self.G3_bias)) * 255).astype(int)

        return self.Gout

    def calc_disc_grad(self, label, batch_size):
        """
        
        """
        dV_dOut = ((label / self.Dout) - ((1 - label) / (1 - self.Dout))) / batch_size
        deltaOut = dV_dOut * sigmoid_prime(np.matmul(self.Theta4, self.D3_bias))
        delta3 = np.matmul(self.Theta4.T, deltaOut)[1:] * relu_prime(np.matmul(self.Theta3, self.D2_bias))
        delta2 = np.matmul(self.Theta3.T, delta3)[1:] * relu_prime(np.matmul(self.Theta2, self.D1_bias))
        delta1 = np.matmul(self.Theta2.T, delta2)[1:] * relu_prime(np.matmul(self.Theta1, self.Din_bias))
        
        dV_dTheta4 = np.matmul(deltaOut, self.D3_bias.T)
        dV_dTheta3 = np.matmul(delta3, self.D2_bias.T)
        dV_dTheta2 = np.matmul(delta2, self.D1_bias.T)
        dV_dTheta1 = np.matmul(delta1, self.Din_bias.T)

        self._Delta1 = self.Delta1 + dV_dTheta1
        self._Delta2 = self.Delta2 + dV_dTheta2
        self._Delta3 = self.Delta3 + dV_dTheta3
        self._Delta4 = self.Delta4 + dV_dTheta4
        
    def calc_gen_grad(self, batch_size):
        """
        
        """
        dV_dOut = (1 / self.Dout) / batch_size
        deltaOut = dV_dOut * sigmoid_prime(np.matmul(self.Theta4, self.D3_bias))
        delta3 = np.matmul(self.Theta4.T, deltaOut)[1:] * relu_prime(np.matmul(self.Theta3, self.D2_bias))
        delta2 = np.matmul(self.Theta3.T, delta3)[1:] * relu_prime(np.matmul(self.Theta2, self.D1_bias))
        delta1 = np.matmul(self.Theta2.T, delta2)[1:] * relu_prime(np.matmul(self.Theta1, self.Din_bias))

        gammaOut = np.matmul(self.Theta1.T, delta1)[1:] * 255 * sigmoid_prime(np.matmul(self.Phi4, self.G3_bias))
        gamma3 = np.matmul(self.Phi4.T, gammaOut)[1:] * relu_prime(np.matmul(self.Phi3, self.G2_bias))
        gamma2 = np.matmul(self.Phi3.T, gamma3)[1:] * relu_prime(np.matmul(self.Phi2, self.G1_bias))
        gamma1 = np.matmul(self.Phi2.T, gamma2)[1:] * relu_prime(np.matmul(self.Phi1, self.Gin_bias))

        dV_dPhi4 = np.matmul(gammaOut, self.G3_bias.T)
        dV_dPhi3 = np.matmul(gamma3, self.G2_bias.T)
        dV_dPhi2 = np.matmul(gamma2, self.G1_bias.T)
        dV_dPhi1 = np.matmul(gamma1, self.Gin_bias.T)
        
        self._Gamma1 = self.Gamma1 + dV_dPhi1
        self._Gamma2 = self.Gamma2 + dV_dPhi2
        self._Gamma3 = self.Gamma3 + dV_dPhi3
        self._Gamma4 = self.Gamma4 + dV_dPhi4

    def update_Thetas(self):
        """
        
        """
        self._Theta1 = self.Theta1 + self._lambda_Theta1 * self.Delta1
        self._Theta2 = self.Theta2 + self._lambda_Theta2 * self.Delta2
        self._Theta3 = self.Theta3 + self._lambda_Theta3 * self.Delta3
        self._Theta4 = self.Theta4 + self._lambda_Theta4 * self.Delta4

        self._Delta1 = np.zeros(self.Theta1.shape)
        self._Delta2 = np.zeros(self.Theta2.shape)
        self._Delta3 = np.zeros(self.Theta3.shape)
        self._Delta4 = np.zeros(self.Theta4.shape)
    
    def update_Phis(self):
        """
        
        """
        self._Phi1 = self.Phi1 + self._lambda_Phi1 * self.Gamma1
        self._Phi2 = self.Phi2 + self._lambda_Phi2 * self.Gamma2
        self._Phi3 = self.Phi3 + self._lambda_Phi3 * self.Gamma3
        self._Phi4 = self.Phi4 + self._lambda_Phi4 * self.Gamma4

        self._Gamma1 = np.zeros(self.Phi1.shape)
        self._Gamma2 = np.zeros(self.Phi2.shape)
        self._Gamma3 = np.zeros(self.Phi3.shape)
        self._Gamma4 = np.zeros(self.Phi4.shape)

