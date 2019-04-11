from scipy.special import comb
from sympy import *
from numpy import linalg as LA
from numpy import pi, tanh, array, dot
from scipy.optimize import linprog

import numpy as np
import sympy as sp
import itertools
import math


def nn_poly_approx_bernstein(f, state_vars, d, box, output_index):
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
    poly_min = math.inf
    poly_max = -math.inf
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
        monomial = f(np.array(point))[output_index]
        if monomial < poly_min:
            poly_min = monomial
        if monomial > poly_max:
            poly_max = monomial
        for j in range(m):
            y_j = y[j]
            k_j = cb[j]
            d_j = d[j]
            monomial = monomial*round(comb(d_j,k_j))*(y_j**k_j)*((1-y_j)**(d_j-k_j))
        bernstein = bernstein + monomial
    # print(p2c(bernstein))
    # construct polynomial approximation for the overall controller based on bernstein polynomial
    poly_approx = bernstein[0]
    for j in range(m):
        y_j = y[j]
        x_j = x[j]
        alpha_j = box[j][0]
        beta_j = box[j][1]
        poly_approx = poly_approx.subs(y_j, (x_j-alpha_j)/(beta_j-alpha_j))
    return simplify(poly_approx), poly_min[0], poly_max[0]

def bernstein_error(f_details, f, d, box, output_index, activation):
    lips, network_output_range = lipschitz(f_details, box, output_index, activation)
    print('Lipschitz constant: {}'.format(lips))

    m = len(d)
    error_bound_lips = lips/2
    temp = 0
    for j in range(m):
        d_j = d[j]
        temp = temp + 1/d_j
        # lower bound of the j-th component
        alpha_j = box[j][0]
        # upper bound of the j-th component
        beta_j = box[j][1]
        error_bound_lips = error_bound_lips * (beta_j-alpha_j)
    error_bound_lips = error_bound_lips * math.sqrt(temp)


    x = sp.symbols('x:'+ str(f_details.num_of_inputs))
    b, poly_min, poly_max = nn_poly_approx_bernstein(f, x, d, box, output_index)
    error_bound_interval = max([poly_min-network_output_range[0][0][0], network_output_range[0][1][0]-poly_max, 0])

    print('network_output_range: {}'.format(network_output_range[0]))
    print('poly_range: {}'.format([poly_min, poly_max]))
    print('error_bound_lips: {}'.format(error_bound_lips))
    print('error_bound_interval: {}'.format(error_bound_interval))
    return min([error_bound_lips, error_bound_interval])


##############################################################
def lipschitz(NN_controller, network_input_box, output_index, activation):
    weight_all_layer = NN_controller.weights
    bias_all_layer = NN_controller.bias
    offset = NN_controller.offset
    scale_factor = NN_controller.scale_factor

    layers = len(bias_all_layer)
    lips = 1
    input_range_layer = network_input_box
    for j in range(layers):
        if j < layers - 1:
            weight_j = weight_all_layer[j]
        else:
            weight_j = np.reshape(weight_all_layer[j][output_index], (1, -1))
        if j < layers - 1:
            bias_j = bias_all_layer[j]
        else:
            bias_j = np.reshape(bias_all_layer[j][output_index], (1, -1))
        lipschitz_j = lipschitz_layer(weight_j, bias_j, input_range_layer, activation)
        lips = lips * lipschitz_j
        input_range_layer, _ = output_range_layer(weight_j, bias_j, input_range_layer, activation)
    return lips* scale_factor, (np.array(input_range_layer)-offset)* scale_factor

def lipschitz_layer(weight, bias, input_range_layer, activation):
    neuron_dim = bias.shape[0]
    output_range_box, new_weight = output_range_layer(weight, bias, input_range_layer, activation)
    if activation == 'ReLU':
        return LA.norm(new_weight, 2)
    if activation == 'sigmoid':
        max_singular = 0
        for j in range(neuron_dim):
            range_j = output_range_box[j]
            if range_j[0] > 0.5:
                singular_j = range_j[0]*(1-range_j[0])
            elif range_j[1] < 0.5:
                singular_j = range_j[1]*(1-range_j[1])
            else:
                singular_j = np.array([0.25])
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
                singular_j = np.array([1])
            if max_singular < singular_j:
                max_singular = singular_j
        return max_singular*LA.norm(weight, 2)

def output_range_layer(weight, bias, input_range_layer, activation):
    # solving LPs
    neuron_dim = bias.shape[0]
    output_range_box = []
    new_weight = []
    for j in range(neuron_dim):
        # c: weight of the j-th dimension
        c = weight[j]
        c = c.transpose()
        #print('c: ' + str(c))
        b = bias[j]
        #print('b: ' + str(b))
        # compute the minimal input
        res_min = linprog(c, bounds=input_range_layer, options={"disp": False})
        input_j_min = res_min.fun + b
        #print('min: ' + str(input_j_min))
        # compute the minimal output
        if activation == 'ReLU':
            if input_j_min < 0:
                output_j_min = np.array([0])
            else:
                output_j_min = input_j_min
        if activation == 'sigmoid':
            output_j_min = 1/(1+np.exp(-input_j_min))
        if activation == 'tanh':
            output_j_min = 2/(1+np.exp(-2*input_j_min))-1
        # compute the maximal input
        res_max = linprog(-c, bounds=input_range_layer, options={"disp": False})
        input_j_max = -res_max.fun + b
        # compute the maximal output
        if activation == 'ReLU':
            if input_j_max < 0:
                output_j_max = np.array([0])
            else:
                output_j_max = input_j_max
                new_weight.append(weight[j])
        if activation == 'sigmoid':
            output_j_max = 1/(1+np.exp(-input_j_max))
        if activation == 'tanh':
            output_j_max = 2/(1+np.exp(-2*input_j_max))-1
        output_range_box.append([output_j_min, output_j_max])
    return output_range_box, new_weight


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



