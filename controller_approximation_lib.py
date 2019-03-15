#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
import ast
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot


def poly_approx_controller(d_str, box_str, output_index):
    NN_controller = nn_controller_details()
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
    b = bp.nn_poly_approx_bernstein(nn_controller(), x, d, box, output_i)
    return bp.p2c(b)


def poly_approx_error(lips_str, d_str, box_str):
    lips = ast.literal_eval(lips_str)
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    error_bound = bp.bernstein_error(lips, d, box)
    return bp.p2c(error_bound)

def network_lips(box_str, activation):
    NN_controller = nn_controller_details()
    box = ast.literal_eval(box_str)
    lips = bp.lipschitz(NN_controller.weights, NN_controller.bias, box, activation)*NN_controller.scale_factor
    return bp.p2c(lips)
