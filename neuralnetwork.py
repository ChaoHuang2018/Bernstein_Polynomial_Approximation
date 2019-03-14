from numpy import pi, tanh, array, dot
import numpy as np


beta = 0.1


class NN:
    """
    A Neural Network with one hidden layer with tanh as activation function
    """
    def __init__(self, x, lambda0=0, lambda1=0):
        """
        x: input parameters of the NN
        lambda0: desired norm of weight0 matrix
        lambda1: desired norm of weight1 matrix
        """
        # Compute the Lipschitz constant bound
        # Num of hidden layer neurons
        h = round((len(x)-1)/4)
        # Weights0 Matrix
        weights0 = np.reshape(array(x[h:3*h]), (2, h))
        if lambda0 != 0:
            K0 = np.linalg.norm(weights0, 2)/lambda0
        else:
            K0 = 1
        # Weights1 Matrix
        weights1 = x[0:h]
        if lambda1 != 0:
            K1 = np.linalg.norm(weights1, 2)/lambda1
        else:
            K1 = 1
        # Parseval Regularization
        self.weights0 = ((1.0 + beta) * weights0
        - beta * np.matmul(np.matmul(weights0, weights0.T), weights0))/K0

        self.weights1 = ((1.0 + beta) * weights1
        - beta * np.dot(weights1, weights1) * weights1)/K1

        # Weights and bias for the hidden layer
        # self.weight_0_d = x[h:2*h]/(max(1, K0))
        # self.weight_0_t = x[2*h:3*h]/(max(1, K0))
        self.weight_0_d = self.weights0[0]
        self.weight_0_t = self.weights0[1]
        self.bias_0 = x[3*h:4*h]
        # Weights and bias for the output layer
        # self.weight_1 = x[0:h]/(max(1, K1))
        self.weight_1 = weights1
        self.bias_1 = x[4*h]

    def controller(self, x):
        """
        Feedforward computation

        output: [-1, 1]
        """
        d_err = x[0]
        t_err = x[1]
        output = tanh(dot(self.weight_1, tanh(
            (self.weight_0_d * d_err)+(
                self.weight_0_t * t_err))+self.bias_0) + self.bias_1)

        return pi*(output + 1)

    def Lipschitz_constant(self):
        K0 = np.linalg.norm(self.weights0, 2)
        K1 = np.linalg.norm(self.weights1, 2)
        K = K0 * K1
        return K0, K1, K

    def weight(self):
        network_weight = []
        #weight_0 = [self.weight_0_d, self.weight_0_t]
        #weight_1 = self.weight_1
        weight_0 = self.weights0
        network_weight.append(weight_0.transpose())
        network_weight.append(self.weights1)
        #network_weight.append([pi])
        return network_weight

    def bias(self):
        network_bias = []
        bias_0 = self.bias_0
        bias_1 = self.bias_1
        network_bias.append(bias_0)
        network_bias.append([bias_1])
        #network_bias.append([pi])
        return network_bias
        
