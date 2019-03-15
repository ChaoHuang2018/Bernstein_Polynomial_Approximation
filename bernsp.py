from scipy.special import comb
from sympy import *
from numpy import linalg as LA
from numpy import pi, tanh, array, dot
from scipy.optimize import linprog

import numpy as np
import sympy as sp
import itertools
import math


def nn_poly_approx_bernstein(f, state_vars, d, box):
    """
    bernstein polynomial approximation of a given function f on a general box space
    f: a function
    state_var: the input variable of f
    d: degree bound vector of bernstein polynomial
    box: box space of state variables [\alpha_1,\beta_1]\times \cdots \times [\alpha_m,\beta_m]
    """
    m = len(state_vars)
    x = state_vars
    all_comb_lists = degree_comb_lists(d,m)
    bernstein = 0
    # construct bernstein polynomial for recover function + nerual network
    y = sp.symbols('y:'+str(m))
    for cb in all_comb_lists:
        point = [];
        for j in range(m):
            k_j = cb[j]
            d_j = d[j]
            # linear transformation to normalize the box to I=[0,1]^m
            # lower bound of the j-th component
            alpha_j = float(box[j][0])
            # upper bound of the j-th component
            beta_j = float(box[j][1])
            point.append((beta_j-alpha_j)*(cb[j]/d[j])+alpha_j)
        monomial = f(np.array(point))
        for j in range(m):
            y_j = y[j]
            k_j = cb[j]
            d_j = d[j]
            monomial = monomial*round(comb(d_j,k_j))*(y_j**k_j)*((1-y_j)**(d_j-k_j))
        bernstein = bernstein + monomial
    # print(p2c(bernstein))
    # construct polynomial approximation for the overall controller based on bernstein polynomial
    poly_approx = bernstein
    for j in range(m):
        y_j = y[j]
        x_j = x[j]
        alpha_j = box[j][0]
        beta_j = box[j][1]
        poly_approx = poly_approx.subs(y_j, (x_j-alpha_j)/(beta_j-alpha_j))
    return simplify(poly_approx)
                        
def bernstein_error(lips, d, box):
    m = len(d)
    error_bound = lips/2
    temp = 0
    for j in range(m):
        d_j = d[j]
        temp = temp + 1/d_j
        # lower bound of the j-th component
        alpha_j = box[j][0]
        # upper bound of the j-th component
        beta_j = box[j][1]
        error_bound = error_bound * (beta_j-alpha_j)
    error_bound = error_bound * math.sqrt(temp)
    return error_bound


##############################################################
def lipschitz(weight_all_layer, bias_all_layer, network_input_box, activation):
    layers = len(bias_all_layer)
    lips = 1
    input_range_layer = network_input_box
    for j in range(layers):
        weight_j = weight_all_layer[j]
        bias_j = bias_all_layer[j]
        lipschitz_j = lipschitz_layer(weight_j, bias_j, input_range_layer, activation)
        lips = lips * lipschitz_j
        input_range_layer = output_range_layer(weight_j, bias_j, input_range_layer, activation)
    return lips

def lipschitz_layer(weight, bias, input_range_layer, activation):
    neuron_dim = bias.shape[0]
    output_range_box = output_range_layer(weight, bias, input_range_layer, activation)
    if activation == 'ReLU':
        return LA.norm(weight, 2)
    if activation == 'sigmoid':
        max_singular = 0
        for j in range(neuron_dim):
            range_j = output_range_box[j]
            if range_j[0] > 0.5:
                singular_j = range_j[0]*(1-range_j[0])
            elif range_j[1] < 0.5:
                singular_j = range_j[1]*(1-range_j[1])
            else:
                singular_j = 0.25
            if max_singular < singular_j:
                max_singular = singular_j
        return max_singular*LA.norm(weight, 2)
    if activation == 'tanh':
        max_singular = 0
        for j in range(neuron_dim):
            range_j = output_range_box[j]
            if range_j[0] > 0:
                singular_j = 1 - range_j[0]**2
            elif range_j[1] < 0:
                singular_j = 1 - range_j[1]**2
            else:
                singular_j = 1
            if max_singular < singular_j:
                max_singular = singular_j
        return max_singular*LA.norm(weight, 2)

def output_range_layer(weight, bias, input_range_layer, activation):
    # solving LPs
    neuron_dim = bias.shape[0]
    output_range_box = []
    for j in range(neuron_dim):
        # c: weight of the j-th dimension
        c = weight[j]
        c = c.transpose()
        b = bias[j]
        # compute the minimal input 
        res_min = linprog(c, bounds=input_range_layer, options={"disp": False})
        input_j_min = res_min.fun + b
        # compute the minimal output 
        if activation == 'ReLU':
            if input_j_min < 0:
                output_j_min = 0
            else:
                output_j_min = input_j_min
        if activation == 'sigmoid':
            output_j_min = 1/(1+math.exp(input_j_min))
        if activation == 'tanh':
            output_j_min = 2/(1+math.exp(-2*input_j_min))-1
        # compute the maximal input
        res_max = linprog(-c, bounds=input_range_layer, options={"disp": False})
        input_j_max = -res_max.fun + b
        # compute the maximal output 
        if activation == 'ReLU':
            if input_j_max < 0:
                output_j_max = 0
            else:
                output_j_max = input_j_max
        if activation == 'sigmoid':
            output_j_max = 1/(1+math.exp(input_j_max))
        if activation == 'tanh':
            output_j_max = 2/(1+math.exp(-2*input_j_max))-1
        output_range_box.append([output_j_min, output_j_max])
    return output_range_box
        

##############################################################
def degree_comb_lists(d, m):
    # generate the degree combination list
    degree_lists = []
    for j in range(m):
        degree_lists.append(range(d[j]+1))
    all_comb_lists = list(itertools.product(*degree_lists))
    return all_comb_lists

def p2c(py_b):
    str_b = str(py_b)
    c_b = str_b.replace("**", "^")
    return c_b


# a simple test case
def test_f(x):
    return math.sin(x[0])+math.cos(x[1])

    

