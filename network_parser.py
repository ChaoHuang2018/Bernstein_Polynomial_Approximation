import numpy as np
from neuralnetwork_relu import NN

FILENAME = 'neural_network_controller'
OFFSET = 4
SCALE_FACTOR = 1


def nn_controller():
    """
    Return the network controller function
    """
    # Obtain the trained parameters and assign the value to res
    with open(FILENAME) as inputfile:
        lines = inputfile.readlines()
    length = len(lines)
    res = np.zeros(length)
    for i, text in enumerate(lines):
        res[i] = eval(text)

    # Set the controller
    NN_controller = NN(res, OFFSET, SCALE_FACTOR)
    controller = NN_controller.controller
    return controller

def nn_controller_details():
    """
    Return weights and bias
    """
    with open(FILENAME) as inputfile:
        lines = inputfile.readlines()
    length = len(lines)
    res = np.zeros(length)
    for i, text in enumerate(lines):
        res[i] = eval(text)

    # Set the controller
    NN_controller = NN(res, OFFSET, SCALE_FACTOR)

    return NN_controller
