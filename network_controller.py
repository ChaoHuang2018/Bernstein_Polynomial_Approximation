import sys
import numpy as np
import os
from neuralnetwork import NN
import json
import bernsp as bp

FILENAME = 'param.json'


def nn_controller():
    # Obtain the trained parameters and assign the value to res
    # searchfile = open(FILENAME)
    # lines = searchfile.readlines()
    # res = np.array(eval(lines[0]))
    # searchfile.close()
    with open(FILENAME) as inputfile:
        res = np.array(json.load(inputfile))
    # Set the controller
    NN_controller = NN(res, lambda0=4, lambda1=0)
    controller = NN_controller.controller
    return controller


def nn_controller_details():
    # Obtain the trained parameters and assign the value to res
    # searchfile = open(FILENAME)
    # lines = searchfile.readlines()
    # res = np.array(eval(lines[0]))
    # searchfile.close()
    with open(FILENAME) as inputfile:
        res = np.array(json.load(inputfile))
    # Set the controller
    NN_controller = NN(res, lambda0=4, lambda1=0)
    weight = NN_controller.weight()
    bias = NN_controller.bias()
    return weight, bias


