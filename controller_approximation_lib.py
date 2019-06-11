#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
import ast
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot


def poly_approx_controller(d_str, box_str, output_index, activation, nerual_network):
    NN_controller = nn_controller_details(nerual_network, activation)
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
    b, _ , _ = bp.nn_poly_approx_bernstein(nn_controller(nerual_network, activation), x, d, box, output_i)
    return bp.p2c(b)


def poly_approx_error(d_str, box_str, output_index, activation, nerual_network, num_partition):
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    num_partition = ast.literal_eval(num_partition)
    # error_bound = bp.bernstein_error(nn_controller_details(nerual_network, activation), nn_controller(nerual_network, activation), d, box, output_i, activation, nerual_network)
    error_bound = bp.bernstein_error_partition(nn_controller_details(nerual_network, activation), nn_controller(nerual_network, activation), d, box, output_i, activation, nerual_network, num_partition)
    return bp.p2c(error_bound)

def network_lips(box_str, activation):
    NN_controller = nn_controller_details()
    box = ast.literal_eval(box_str)
    lips, _ = bp.lipschitz(NN_controller, box, activation)
    return bp.p2c(lips)

def network_output_range_center(d_str, box_str, output_index, activation, nerual_network):
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    _, output_range = bp.lipschitz(nn_controller_details(nerual_network, activation), box, output_i, activation)
    return bp.p2c((output_range[0][0][0]+output_range[0][1][0])/2)

def network_output_range_radius(d_str, box_str, output_index, activation, nerual_network):
    box = ast.literal_eval(box_str)
    output_i = ast.literal_eval(output_index)
    _, output_range = bp.lipschitz(nn_controller_details(nerual_network, activation), box, output_i, activation)
    return bp.p2c((output_range[0][1][0]-output_range[0][0][0])/2)
