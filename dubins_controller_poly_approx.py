#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
import ast
from network_controller import dubins_car_nn_controller, dubins_car_nn_controller_details
from numpy import pi, tanh, array, dot


def dubins_poly_controller(d_str, box_str):
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    x = ['d_err','t_err']
    b = bp.nn_poly_approx_bernstein(dubins_car_nn_controller(), x, d, box)
    return bp.p2c(b)


def poly_approx_error(lips_str, d_str, box_str):
    lips = ast.literal_eval(lips_str)
    d = ast.literal_eval(d_str)
    box = ast.literal_eval(box_str)
    error_bound = bp.bernstein_error(lips, d, box)
    return bp.p2c(error_bound)

def network_lips(box_str, activation):
    weight, bias = dubins_car_nn_controller_details()
    box = ast.literal_eval(box_str)
    lips = bp.lipschitz(weight, bias, box, activation)*pi
    return bp.p2c(lips)
