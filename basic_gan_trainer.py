"""
This module contains the GAN trainer class that has the functionality to
initialize random parameters for a new GAN and the appropriate methods for 
training that GAN.

Specifically, the GAN is composed of two multilayer perceptrons of fixed size.
The generator has six layers of size 101, 250, 350, 500, 500, 784, respectively.
The discriminator has five layers of size 785, 500, 250, 50, 1, respectively.
This GAN trainer class has methods to complete a forward pass through the MLPs,
calculate the gradient with respect to all of the weights, and track the value
function over the course of training for the generator and discriminator.
"""
import numpy as np 
from activation_functions import leaky_relu, leaky_relu_prime

class BasicGANTrainer:
    """
    A GAN trainer that creates a new GAN made of multi-layer perceptrons on
    initialization and trains it using backpropagation.

    Attributes:
        _current_disc_value: a float that represents the value of the
            discriminator's value function over a given batch
        _current_gen_value: a float that represents the value of the
            generator's value function over a given batch

        _disc_value_array: a numpy array that accumulates a new
            _current_disc_value after each batch for long-term storage
        _gen_value_array: a numpy array that accumulates a new
            _current_gen_value after each batch for long-term storage

        _clip_grad: a float representing the value at which to clip the gradient

        _lambda_Theta4: a float representing the learning rate for Theta4
        _lambda_Theta3: a float representing the learning rate for Theta3
        _lambda_Theta2: a float representing the learning rate for Theta2
        _lambda_Theta1: a float representing the learning rate for Theta1

        _lambda_Phi5: a float representing the learning rate for Phi5
        _lambda_Phi4: a float representing the learning rate for Phi4
        _lambda_Phi3: a float representing the learning rate for Phi3
        _lambda_Phi2: a float representing the learning rate for Phi2
        _lambda_Phi1: a float representing the learning rate for Phi1

        _Din_bias: a numpy array containing the most recent input layer to the 
            discriminator, including the bias term
        _D1_bias: a numpy array containing the most recent D1 (first hidden
            layer) of the discriminator, including the bias term
        _D2_bias: a numpy array containing the most recent D2 (second hidden
            layer) of the discriminator, including the bias term
        _D3_bias: a numpy array containing the most recent D3 (third hidden
            layer) of the discriminator, including the bias term
        _Dout: a numpy array containing the most recent output layer of the 
            discriminator, including the bias term
        _Theta1: a numpy array containing the weights between the input layer
            and the first hidden layer of the discriminator
        _Theta2: a numpy array containing the weights between the first hidden
            layer and the second hidden layer of the discriminator
        _Theta3: a numpy array containing the weights between the second hidden
            layer and the third hidden layer of the discriminator
        _Theta4: a numpy array containing the weights between the third hidden
            layer and the output layer of the discriminator
        _Delta1: a numpy array that is the same size as Theta1 that accumulates
            the gradient of Theta1 over a given batch - used to update Theta1
            and then reset after each batch
        _Delta2: a numpy array that is the same size as Theta2 that accumulates
            the gradient of Theta2 over a given batch - used to update Theta2
            and then reset after each batch
        _Delta3: a numpy array that is the same size as Theta3 that accumulates
            the gradient of Theta3 over a given batch - used to update Theta3
            and then reset after each batch
        _Delta4: a numpy array that is the same size as Theta4 that accumulates
            the gradient of Theta4 over a given batch - used to update Theta4
            and then reset after each batch

        _Gin_bias: a numpy array containing the most recent input layer to the 
            generator, including the bias term
        _G1_bias: a numpy array containing the most recent G1 (first hidden
            layer) of the generator, including the bias term
        _G2_bias: a numpy array containing the most recent G2 (second hidden
            layer) of the generator, including the bias term
        _G3_bias: a numpy array containing the most recent G3 (third hidden
            layer) of the generator, including the bias term
        _G4_bias: a numpy array containing the most recent G4 (fourth hidden
            layer) of the generator, including the bias term
        _Gout: a numpy array containing the most recent output layer of the 
            generator, including the bias term
        _Phi1: a numpy array containing the weights between the input layer and
            the first hidden layer of the generator
        _Phi2: a numpy array containing the weights between the first hidden
            layer and the second hidden layer of the generator
        _Phi3: a numpy array containing the weights between the second hidden
            layer and the third hidden layer of the generator
        _Phi4: a numpy array containing the weights between the third hidden
            layer and the fourth hidden layer of the generator
        _Phi5: a numpy array containing the weights between the fourth hidden
            layer and the output layer of the generator
        _Gamma1: a numpy array that is the same size as Phi1 that accumulates
            the gradient of Phi1 over a given batch - used to update Phi1
            and then reset after each batch
        _Gamma2: a numpy array that is the same size as Phi2 that accumulates
            the gradient of Phi2 over a given batch - used to update Phi2
            and then reset after each batch
        _Gamma3: a numpy array that is the same size as Phi3 that accumulates
            the gradient of Phi3 over a given batch - used to update Phi3
            and then reset after each batch
        _Gamma4: a numpy array that is the same size as Phi4 that accumulates
            the gradient of Phi4 over a given batch - used to update Phi4
            and then reset after each batch
        _Gamma4: a numpy array that is the same size as Phi5 that accumulates
            the gradient of Phi5 over a given batch - used to update Phi5
            and then reset after each batch
    """

    def __init__(self):
        """
        Set the value of the gen. and  disc. to 0, create arrays to track the 
        value over the course of training, set the learning rates, randomly 
        initialize weights, and set up array to accumulate gradient.

        Args:
            None
        Returns:
            No Return Value
        """
        # Accumulators for disc. and gen. value functions over training batch
        self._current_disc_value = 0
        self._current_gen_value = 0

        # Long-term storage of disc. and gen. value functions per batch
        self._disc_value_array = np.array([])
        self._gen_value_array = np.array([])

        # Value to clip gradient at
        self._clip_grad = .1

        # Layer-specific learning rates
        self._lambda_Theta4 = .000001
        self._lambda_Theta3 = .00001
        self._lambda_Theta2 = .0001
        self._lambda_Theta1 = .0001

        self._lambda_Phi5 = .000001
        self._lambda_Phi4 = .000001
        self._lambda_Phi3 = .00001
        self._lambda_Phi2 = .0001
        self._lambda_Phi1 = .0001

        # Setup disc. layers and random weights
        self._Din_bias = None
        self._D1_bias = None
        self._D2_bias = None
        self._D3_bias = None
        self._Dout = None
        self._Theta1 = (np.random.rand(499, 785) - .5) / 10
        self._Theta2 = (np.random.rand(249, 500) - .5) / 10
        self._Theta3 = (np.random.rand(49, 250) - .5) / 10
        self._Theta4 = (np.random.rand(1, 50) - .5) / 10
        self._Delta1 = np.zeros(self.Theta1.shape)
        self._Delta2 = np.zeros(self.Theta2.shape)
        self._Delta3 = np.zeros(self.Theta3.shape)
        self._Delta4 = np.zeros(self.Theta4.shape)

        # set up gen. layers and random weights
        self._Gin_bias = None
        self._G1_bias = None
        self._G2_bias = None
        self._G3_bias = None
        self._G4_bias = None
        self._Gout = None
        self._Phi1 = (np.random.rand(249, 101) - .5) / 2
        self._Phi2 = (np.random.rand(349, 250) - .5) / 2
        self._Phi3 = (np.random.rand(499, 350) - .5) / 2
        self._Phi4 = (np.random.rand(499, 500) - .5) / 2
        self._Phi5 = (np.random.rand(784, 500) - .5) / 2
        self._Gamma1 = np.zeros(self.Phi1.shape)
        self._Gamma2 = np.zeros(self.Phi2.shape)
        self._Gamma3 = np.zeros(self.Phi3.shape)
        self._Gamma4 = np.zeros(self.Phi4.shape)
        self._Gamma5 = np.zeros(self.Phi5.shape)

    @property 
    def clip_grad(self):
        """
        A property to return _clip_grad
        """
        return self._clip_grad
    
    @property
    def current_disc_value(self):
        """
        A property to return _current_disc_value
        """
        return self._current_disc_value

    @property
    def current_gen_value(self):
        """
        A property to return _current_gen_value
        """
        return self._current_gen_value

    @property
    def disc_value_array(self):
        """
        A property to return _disc_value_array
        """
        return self._disc_value_array

    @property
    def gen_value_array(self):
        """
        A property to return _gen_value_array
        """
        return self._gen_value_array
    
    @property
    def lambda_Theta1(self):
        """
        A property to return _lambda_Theta1
        """
        return self._lambda_Theta1

    @property
    def lambda_Theta2(self):
        """
        A property to return _lambda_Theta2
        """
        return self._lambda_Theta2

    @property
    def lambda_Theta3(self):
        """
        A property to return _lambda_Theta3
        """
        return self._lambda_Theta3

    @property
    def lambda_Theta4(self):
        """
        A property to return _lambda_Theta4
        """
        return self._lambda_Theta4

    @property
    def lambda_Phi1(self):
        """
        A property to return _lambda_Phi1
        """
        return self._lambda_Phi1

    @property
    def lambda_Phi2(self):
        """
        A property to return _lambda_Phi2
        """
        return self._lambda_Phi2

    @property
    def lambda_Phi3(self):
        """
        A property to return _lambda_Phi3
        """
        return self._lambda_Phi3

    @property
    def lambda_Phi4(self):
        """
        A property to return _lambda_Phi4
        """
        return self._lambda_Phi4

    @property
    def lambda_Phi5(self):
        """
        A property to return _lambda_Phi5
        """
        return self._lambda_Phi5

    @property
    def Din_bias(self):
        """
        A property to return _Din_bias
        """
        return self._Din_bias

    @property
    def D1_bias(self):
        """
        A property to return _D1_bias
        """
        return self._D1_bias

    @property
    def D2_bias(self):
        """
        A property to return _D2_bias
        """
        return self._D2_bias

    @property
    def D3_bias(self):
        """
        A property to return _D3_bias
        """
        return self._D3_bias

    @property
    def Dout(self):
        """
        A property to return _Dout_bias
        """
        return self._Dout

    @property
    def Gin_bias(self):
        """
        A property to return _Gin_bias
        """
        return self._Gin_bias

    @property
    def G1_bias(self):
        """
        A property to return _G1_bias
        """
        return self._G1_bias

    @property
    def G2_bias(self):
        """
        A property to return _G2_bias
        """
        return self._G2_bias

    @property
    def G3_bias(self):
        """
        A property to return _G3_bias
        """
        return self._G3_bias

    @property
    def G4_bias(self):
        """
        A property to return _G4_bias
        """
        return self._G4_bias

    @property
    def Gout(self):
        """
        A property to return _Gout_bias
        """
        return self._Gout

    @property
    def Theta1(self):
        """
        A property to return _Theta1
        """
        return self._Theta1

    @property
    def Theta2(self):
        """
        A property to return _Theta2
        """
        return self._Theta2

    @property
    def Theta3(self):
        """
        A property to return _Theta3
        """
        return self._Theta3

    @property
    def Theta4(self):
        """
        A property to return _Theta4
        """
        return self._Theta4

    @property
    def Phi1(self):
        """
        A property to return _Phi1
        """
        return self._Phi1

    @property
    def Phi2(self):
        """
        A property to return _Phi2
        """
        return self._Phi2

    @property
    def Phi3(self):
        """
        A property to return _Phi3
        """
        return self._Phi3

    @property
    def Phi4(self):
        """
        A property to return _Phi4
        """
        return self._Phi4

    @property
    def Phi5(self):
        """
        A property to return _Phi5
        """
        return self._Phi5

    @property
    def Delta1(self):
        """
        A property to return _Delta1
        """
        return self._Delta1

    @property
    def Delta2(self):
        """
        A property to return _Delta2
        """
        return self._Delta2

    @property
    def Delta3(self):
        """
        A property to return _Delta3
        """
        return self._Delta3

    @property
    def Delta4(self):
        """
        A property to return _Delta4
        """
        return self._Delta4

    @property
    def Gamma1(self):
        """
        A property to return _Gamma1
        """
        return self._Gamma1

    @property
    def Gamma2(self):
        """
        A property to return _Gamma2
        """
        return self._Gamma2

    @property
    def Gamma3(self):
        """
        A property to return _Gamma3
        """
        return self._Gamma3

    @property
    def Gamma4(self):
        """
        A property to return _Gamma4
        """
        return self._Gamma4

    @property
    def Gamma5(self):
        """
        A property to return _Gamma5
        """
        return self._Gamma5
    
    def disc_forward_pass(self, train_ex):
        """
        Perform a forward pass through the discriminator using a given input.

        Args:
            train_ex: a 784x1 numpy array with a training example
        Returns:
            Dout: a 1x1 numpy array with the value of the output layer
        """
        # Insert a bias term to the front of the input
        self._Din_bias = np.insert(train_ex, 0, 1, axis=0)

        # Calculate the first hidden layer and insert a bias term to the front
        D1 = leaky_relu(self.Theta1 @ self.Din_bias)
        self._D1_bias = np.insert(D1, 0, 1, axis=0)

        # Calculate the second hidden layer and insert a bias term to the front
        D2 = leaky_relu(self.Theta2 @ self.D1_bias)
        self._D2_bias = np.insert(D2, 0, 1, axis=0)

        # Calculate the third hidden layer and insert a bias term to the front
        D3 = leaky_relu(self.Theta3 @ self.D2_bias)
        self._D3_bias = np.insert(D3, 0, 1, axis=0)

        # Calculate the ouput layer
        self._Dout = leaky_relu(self.Theta4 @ self.D3_bias)

        return self.Dout

    def gen_forward_pass(self, train_ex):
        """
        Perform a forward pass through the generator using a given input.

        Args:
            train_ex: a 100x1 numpy array with random noise
        Returns:
            Gout: a 784x1 numpy array with the values of the output layer
        """
        # Insert a bias term to the front of the input
        self._Gin_bias = np.insert(train_ex, 0, 1, axis=0)

        # Calculate the first hidden layer and insert a bias term to the front
        G1 = leaky_relu(self.Phi1 @ self.Gin_bias)
        self._G1_bias = np.insert(G1, 0, 1, axis=0)

        # Calculate the second hidden layer and insert a bias term to the front
        G2 = leaky_relu(self.Phi2 @ self.G1_bias)
        self._G2_bias = np.insert(G2, 0, 1, axis=0)

        # Calculate the third hidden layer and insert a bias term to the front
        G3 = leaky_relu(self.Phi3 @ self.G2_bias)
        self._G3_bias = np.insert(G3, 0, 1, axis=0)

        # Calculate the fourth hidden layer and insert a bias term to the front
        G4 = leaky_relu(self.Phi4 @ self.G3_bias)
        self._G4_bias = np.insert(G4, 0, 1, axis=0)
        
        # Calculate the output layer
        self._Gout = leaky_relu(self.Phi5 @ self.G4_bias)

        return self.Gout

    def calc_disc_grad(self, label, batch_size):
        """
        Calculate the gradient of the discriminator with respect to all of the
        weight arrays after a single forward pass and update the accumulator
        matrices (_Delta1 through _Delta4).

        Args:
            label: an int representing if the most recent pass was a generated
                example or a real example (0 or 1, respectively)
            batch_size: the number of training examples per batch - used to
                scale the gradient
        Returns:
            No Return Value
        """
        # Partial derivative of the value function with respect to the output
        dV_dOut = (label * 1 + (1 - label) * -1) / batch_size

        # Partial derivative of the value function with respect to each hidden 
        # layer using chain rule
        deltaOut = dV_dOut * leaky_relu_prime(self.Theta4 @ self.D3_bias)
        delta3 = (self.Theta4.T @ deltaOut)[1:] * leaky_relu_prime(self.Theta3 @ self.D2_bias)
        delta2 = (self.Theta3.T @ delta3)[1:] * leaky_relu_prime(self.Theta2 @ self.D1_bias)
        delta1 = (self.Theta2.T @ delta2)[1:] * leaky_relu_prime(self.Theta1 @ self.Din_bias)
        
        # Partial derivative of the value function with respect to each of the
        # weight arrays
        dV_dTheta4 = deltaOut @ self.D3_bias.T
        dV_dTheta3 = delta3 @ self.D2_bias.T
        dV_dTheta2 = delta2 @ self.D1_bias.T
        dV_dTheta1 = delta1 @ self.Din_bias.T

        # Update the gradient accumulators
        self._Delta1 = self.Delta1 + dV_dTheta1
        self._Delta2 = self.Delta2 + dV_dTheta2
        self._Delta3 = self.Delta3 + dV_dTheta3
        self._Delta4 = self.Delta4 + dV_dTheta4
        
    def calc_gen_grad(self, batch_size):
        """
        Calculate the gradient of the geenrator with respect to all of the
        weight arrays after a single forward pass and update the accumulator
        matrices (_Gamma1 through _Gamma5).

        Args:
            batch_size: the number of training examples per batch - used to
                scale the gradient
        Returns:
            No Return Value
        """
        # Partial derivative of the value function with respect to the output
        # of the discriminator
        dV_dOut = 1 / batch_size

        # Partial derivatives of the value function with respect to each of the
        # hidden layers using chain rule
        deltaOut = dV_dOut * leaky_relu_prime(self.Theta4 @ self.D3_bias)
        delta3 = (self.Theta4.T @ deltaOut)[1:] * leaky_relu_prime(self.Theta3 @ self.D2_bias)
        delta2 = (self.Theta3.T @ delta3)[1:] * leaky_relu_prime(self.Theta2 @ self.D1_bias)
        delta1 = (self.Theta2.T @ delta2)[1:] * leaky_relu_prime(self.Theta1 @ self.Din_bias)
        
        # Since output of gen. is input of disc., treat them as one layer
        gammaOut = (self.Theta1.T @ delta1)[1:] * leaky_relu_prime(self.Phi5 @ self.G4_bias)
        gamma4 = (self.Phi5.T @ gammaOut)[1:] * leaky_relu_prime(self.Phi4 @ self.G3_bias)
        gamma3 = (self.Phi4.T @ gamma4)[1:] * leaky_relu_prime(self.Phi3 @ self.G2_bias)
        gamma2 = (self.Phi3.T @ gamma3)[1:] * leaky_relu_prime(self.Phi2 @ self.G1_bias)
        gamma1 = (self.Phi2.T @ gamma2)[1:] * leaky_relu_prime(self.Phi1 @ self.Gin_bias)

        # Partial derivatives of the value function with respect to each of the
        # weight arrays
        dV_dPhi5 = gammaOut @ self.G4_bias.T
        dV_dPhi4 = gamma4 @ self.G3_bias.T
        dV_dPhi3 = gamma3 @ self.G2_bias.T
        dV_dPhi2 = gamma2 @ self.G1_bias.T
        dV_dPhi1 = gamma1 @ self.Gin_bias.T
        
        # Update the gradient accumulators
        self._Gamma1 = self.Gamma1 + dV_dPhi1
        self._Gamma2 = self.Gamma2 + dV_dPhi2
        self._Gamma3 = self.Gamma3 + dV_dPhi3
        self._Gamma4 = self.Gamma4 + dV_dPhi4
        self._Gamma5 = self.Gamma5 + dV_dPhi5

    def update_Thetas(self):
        """
        Use the accumulated gradient after each batch to update the weight
        arrays of the discriminator and then reset the gradient accumulators for
        the next batch.

        Args:
            None
        Returns:
            No Return Value
        """
        # Clip gradient prior to updating weights
        self._Delta1[self.Delta1 > self.clip_grad] = self.clip_grad
        self._Delta2[self.Delta2 > self.clip_grad] = self.clip_grad
        self._Delta3[self.Delta3 > self.clip_grad] = self.clip_grad
        self._Delta4[self.Delta4 > self.clip_grad] = self.clip_grad

        self._Delta1[self.Delta1 < - self.clip_grad] = - self.clip_grad
        self._Delta2[self.Delta2 < - self.clip_grad] = - self.clip_grad
        self._Delta3[self.Delta3 < - self.clip_grad] = - self.clip_grad
        self._Delta4[self.Delta4 < - self.clip_grad] = - self.clip_grad
        
        # Scale each gradient by the learning rate of that layer and update the
        # weights
        self._Theta1 = self.Theta1 + self.lambda_Theta1 * self.Delta1
        self._Theta2 = self.Theta2 + self.lambda_Theta2 * self.Delta2
        self._Theta3 = self.Theta3 + self.lambda_Theta3 * self.Delta3
        self._Theta4 = self.Theta4 + self.lambda_Theta4 * self.Delta4

        # Reset gradient accumulators to zero for the next batch
        self._Delta1 = np.zeros(self.Theta1.shape)
        self._Delta2 = np.zeros(self.Theta2.shape)
        self._Delta3 = np.zeros(self.Theta3.shape)
        self._Delta4 = np.zeros(self.Theta4.shape)
    
    def update_Phis(self):
        """
        Use the accumulated gradient after each batch to update the weight
        arrays of the generator and then reset the gradient accumulators for the
        next batch.

        Args:
            None
        Returns:
            No Return Value
        """
        # Clip the gradient prior to updating the weights
        self._Gamma1[self.Gamma1 > self.clip_grad] = self.clip_grad
        self._Gamma2[self.Gamma2 > self.clip_grad] = self.clip_grad
        self._Gamma3[self.Gamma3 > self.clip_grad] = self.clip_grad
        self._Gamma4[self.Gamma4 > self.clip_grad] = self.clip_grad
        self._Gamma5[self.Gamma5 > self.clip_grad] = self.clip_grad

        self._Gamma1[self.Gamma1 < - self.clip_grad] = - self.clip_grad
        self._Gamma2[self.Gamma2 < - self.clip_grad] = - self.clip_grad
        self._Gamma3[self.Gamma3 < - self.clip_grad] = - self.clip_grad
        self._Gamma4[self.Gamma4 < - self.clip_grad] = - self.clip_grad
        self._Gamma5[self.Gamma5 < - self.clip_grad] = - self.clip_grad

        # Sceale the gradient by the learning rate of that layer and update the 
        # weights
        self._Phi1 = self.Phi1 + self.lambda_Phi1 * self.Gamma1
        self._Phi2 = self.Phi2 + self.lambda_Phi2 * self.Gamma2
        self._Phi3 = self.Phi3 + self.lambda_Phi3 * self.Gamma3
        self._Phi4 = self.Phi4 + self.lambda_Phi4 * self.Gamma4
        self._Phi5 = self.Phi5 + self.lambda_Phi5 * self.Gamma5

        # Reset the gradient accumulators to zero for the next batch
        self._Gamma1 = np.zeros(self.Phi1.shape)
        self._Gamma2 = np.zeros(self.Phi2.shape)
        self._Gamma3 = np.zeros(self.Phi3.shape)
        self._Gamma4 = np.zeros(self.Phi4.shape)
        self._Gamma5 = np.zeros(self.Phi5.shape)

    def add_to_disc_value(self, label):
        """
        Add most recent discriminator output to its current value function if it
        was a real example and subtract the most recent discriminator output if
        it was a generated example, per its value function.

        Args:
            label: an int representing the class of the most recent input to the
                discriminator
        Returns:
            No Return Value
        """
        if label == 0:
            self._current_disc_value = self.current_disc_value - self.Dout
        else:
            self._current_disc_value = self.current_disc_value + self.Dout

    def add_to_gen_value(self):
        """
        Given that the most recent input to the discriminator was a generated 
        example, add the most recent discriminator output to the current 
        generator's value function.

        Args:
            None
        Returns:
            No Return Value
        """
        self._current_gen_value = self.Dout

    def update_disc_value_array(self, batch_size):
        """
        After one batch has been completed, append the current discriminator 
        value to the numpy array that stores past values, after scaling by the 
        batch size, and reset the tracked value to zero for the next batch.

        Args:
            batch_size: an int representing the batch size
        Returns:
            No Return Value
        """
        self._disc_value_array = np.append(self.disc_value_array, self.current_disc_value / batch_size)
        self._current_disc_value = 0

    def update_gen_value_array(self, batch_size):
        """
        After one batch has been completed, append the current generator value 
        to to numpy array that stores past values, after scaling by the batch
        size, and reset the tracked value to zero for the next batch.

        Args:
            batch_size: an int representing the batch size
        Returns:
            No Return Value
        """
        self._gen_value_array = np.append(self.gen_value_array, self.current_gen_value / batch_size)
        self._current_gen_value = 0
