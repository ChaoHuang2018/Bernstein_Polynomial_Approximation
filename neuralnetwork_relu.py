import numpy as np


class NN:
    """
    a neural network with relu activation function
    """
    def __init__(self, res, offset, scale_factor):
        # affine mapping of the output
        self.offset = offset
        self.scale_factor = scale_factor

        # parse structure of neural networks
        self.num_of_inputs = int(res[0])
        self.num_of_outputs = int(res[1])
        self.num_of_hidden_layers = int(res[2])
        self.network_structure = np.zeros(self.num_of_hidden_layers + 1,
                                          dtype=int)

        # pointer is current reading index
        self.pointer = 3

        # num of neurons of each layer
        for i in range(self.num_of_hidden_layers):
            self.network_structure[i] = int(res[self.pointer])
            self.pointer += 1

        # output layer
        self.network_structure[-1] = self.num_of_outputs

        # all values from the text file
        self.param = res

        # store the weights and bias in two lists
        # self.weights
        # self.bias
        self.parse_w_b()

    def ReLU(self, x):
        """
        ReLU activation function
        """
        x[x < 0] = 0
        return x

    def parse_w_b(self):
        """
        Parse the input text file
        and store the weights and bias indexed by layer

        Generate: self.weights, self.bias
        """
        # initialize the weights and bias storage space
        self.weights = [None] * (self.num_of_hidden_layers + 1)
        self.bias = [None] * (self.num_of_hidden_layers + 1)

        # compute parameters of the input layer
        weight_matrix0 = np.zeros((self.network_structure[0],
                                   self.num_of_inputs))
        bias_0 = np.zeros((self.network_structure[0], 1))

        for i in range(self.network_structure[0]):
            for j in range(self.num_of_inputs):
                weight_matrix0[i, j] = self.param[self.pointer]
                self.pointer += 1

            bias_0[i] = self.param[self.pointer]
            self.pointer += 1

        # store input layer parameters
        self.weights[0] = weight_matrix0
        self.bias[0] = bias_0

        # compute the hidden layers paramters
        for i in range(self.num_of_hidden_layers):
            weights = np.zeros((self.network_structure[i + 1],
                                self.network_structure[i]))
            bias = np.zeros((self.network_structure[i + 1], 1))

            # read the weight matrix
            for j in range(self.network_structure[i + 1]):
                for k in range(self.network_structure[i]):
                    weights[j][k] = self.param[self.pointer]
                    self.pointer += 1
                bias[j] = self.param[self.pointer]
                self.pointer += 1

            # store parameters of each layer
            self.weights[i + 1] = weights
            self.bias[i + 1] = bias

    def controller(self, x):
        """
        Input: state

        Output: control value after affine transformation
        """
        # transform the input
        length = len(x)
        g = np.array(x, dtype=np.float64).reshape([length, 1])

        # pass input through each layer
        for i in range(self.num_of_hidden_layers + 1):
            # linear transformation
            g = self.weights[i] @ g
            g = g + self.bias[i]

            # activation
            g = self.ReLU(g)

        # affine transformation of output
        y = g - self.offset
        y = y * self.scale_factor

        return y
