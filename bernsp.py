from scipy.special import comb
from sympy import *
from numpy import linalg as LA
from scipy.optimize import linprog
from interval import interval, inf, imath

import numpy as np
import sympy as sp
import itertools
import math


def nn_poly_approx_bernstein(f, state_vars, d, box, output_index):
    """
    bernstein polynomial approximation of a given function f
    on a general box space
    f: a function
    state_var: the input variable of f
    d: degree bound vector of bernstein polynomial
    box: box space of state variables [alpha_1, beta_1] times cdots
    times [alpha_m, beta_m]
    """
    m = len(state_vars)
    x = state_vars
    all_comb_lists = degree_comb_lists(d, m)
    bernstein = 0
    poly_min = np.inf
    poly_max = -np.inf
    # construct bernstein polynomial for recover function + nerual network
    y = sp.symbols('y:'+str(m))
    for cb in all_comb_lists:
        point = []
        for j in range(m):
            k_j = cb[j]
            d_j = d[j]
            # linear transformation to normalize the box to I=[0,1]^m
            # lower bound of the j-th component
            alpha_j = np.float64(box[j][0])
            # upper bound of the j-th component
            beta_j = np.float64(box[j][1])
            point.append((beta_j-alpha_j)*(cb[j]/d[j])+alpha_j)
        monomial = f(np.array(point, dtype=np.float64))[output_index]
        if monomial < poly_min:
            poly_min = monomial
        if monomial > poly_max:
            poly_max = monomial
        for j in range(m):
            y_j = y[j]
            k_j = cb[j]
            d_j = d[j]
            monomial = monomial*comb(d_j, k_j)*(y_j**k_j)*((1-y_j)**(d_j-k_j))
        bernstein = bernstein + monomial
    poly_approx = bernstein[0]
    for j in range(m):
        y_j = y[j]
        x_j = x[j]
        alpha_j = np.float64(box[j][0])
        beta_j = np.float64(box[j][1])
        poly_approx = poly_approx.subs(y_j, (x_j-alpha_j)/(beta_j-alpha_j))
    return poly_approx, poly_min[0], poly_max[0]

steps = -1

def bernstein_error_partition(f_details, f, d, box, output_index, activation, filename, eps):
    global steps
    steps += 1
    m = len(d)
    lips, network_output_range = lipschitz(f_details, box, output_index, activation)

    distance_estimate = 1
    for j in range(m):
        distance_estimate *= abs(np.diff(box[j]))

    LD_estimate = 2 * lips * np.sqrt(m) / 2 ** m * distance_estimate
    num_partition = int(np.ceil((LD_estimate // eps + 1) ** (1/m)))

    partition = [num_partition]*m
    all_comb_lists = degree_comb_lists(partition, m)


    if isinstance(lips, np.ndarray):
        lips = lips[0]

    print('---------------' + filename + '-------------------')
    print('steps: {}'.format(steps))
    print('degree bound: {}'.format(d))
    print('number of partition: {}'.format(num_partition))
    print('Lipschitz constant: {}'.format(lips))

    state_vars = sp.symbols('x:'+ str(m))
    bern, _, _ = nn_poly_approx_bernstein(f, state_vars, d, box, output_index)
    bern_error = bernstein_error(f_details, f, d, box, output_index, activation, filename)

    error = 0
    for cb in all_comb_lists:
        box_temp = []
        for j in range(m):
            k_j = cb[j]
            alpha_j = np.float64(box[j][0])
            beta_j = np.float64(box[j][1])
            box_temp.append([(beta_j-alpha_j)*(cb[j]/num_partition)+alpha_j,(beta_j-alpha_j)*((cb[j]+1)/num_partition)+alpha_j])
        vertex_index_list = degree_comb_lists([1]*m, m)
        for vertex_index in vertex_index_list:
            sample_point = np.zeros(m, dtype=np.float64)
            poly = bern
            distance_m = 1
            for j in range(m):
                sample_point[j] = box_temp[j][vertex_index[j]]
                poly = poly.subs(state_vars[j], sample_point[j])
                distance_m *= abs(np.diff(box_temp[j])[0])
            sample_value = f(sample_point)[output_index]
            sample_diff = abs(np.float64(poly) - sample_value)[0]
            if sample_diff > bern_error and lips != 0.0:
                print('---------------- error ------------------')
                print('box: {}'.format(box))
                print('bern: {}'.format(bern))
                print('box temp: {}'.format(box_temp))
                print('sample point: {}'.format(sample_point))
                print('sample value is {}, poly is {}'.format(sample_value, poly))
                raise ValueError('Sample diff {} is smaller than lip error bound {}'.format(sample_diff, bern_error))
            # print('sample difference: {}'.format(sample_diff))
            # print('LD error: {}'.format(2 * lips * np.sqrt(m) / 2**m * distance_m))
            error_temp = 2 * lips * np.sqrt(m) / 2**m * distance_m + sample_diff
            if error_temp >= error:
                error = error_temp
       # piece_bern, _, _ = nn_poly_approx_bernstein(f, state_vars, d, box_temp, output_index)
        # error_piece_to_NN = bernstein_error(f_details, f, d, box_temp, output_index, activation, filename)

        # error_piece_to_bern = error_functions(bern, piece_bern, d, box_temp, state_vars)

        # print('piece to bern error: {}'.format(error_piece_to_bern))

        # if error_piece_to_NN + error_piece_to_bern >= piecewise_error:
        #     piecewise_error = error_piece_to_NN + error_piece_to_bern
    print('LD error: {}'.format(2 * lips * np.sqrt(m) / 2**m * distance_m))
    print('sample error: {}'.format(error))
    if error > bern_error and bern_error != 0:
        error = bern_error

    if error < np.finfo(np.float64).eps:
        error = 0.0

    return error


def error_functions(f1, f2, d, box, state_vars):
    m = len(d)

    result = {}
    global f_interval
    f_interval = interval(0)
    str_f = str(simplify(f1 - f2))

    for j in range(m):
        str_f = str_f.replace(str(state_vars[j]), 'interval(' + str(box[j]) + ')')

    exec('from interval import interval, inf, imath; f_interval = ' + str_f, result)

    if not isinstance(result['f_interval'], interval):
        error_piece_to_bern = result['f_interval']
    else:
        error_piece_to_bern = abs(result['f_interval'])[0].sup

    return error_piece_to_bern


def bernstein_error(f_details, f, d, box, output_index, activation, filename):
    m = len(d)
    # partition = []
    # num_partition = 14
    # for j in range(m):
    #     partition.append(num_partition)
    # all_comb_lists = degree_comb_lists(partition, m)
    # lips = 0
    # for cb in all_comb_lists:
    #     box_temp = []
    #     for j in range(m):
    #         k_j = cb[j]
    #         alpha_j = np.float64(box[j][0])
    #         beta_j = np.float64(box[j][1])
    #         box_temp.append([(beta_j-alpha_j)*(cb[j]/num_partition)+alpha_j,(beta_j-alpha_j)*((cb[j]+1)/num_partition)+alpha_j])
    #     lips_temp, network_output_range = lipschitz(f_details, box_temp, output_index, activation)
    #     if isinstance(lips_temp, np.ndarray):
    #         if lips_temp[0] >= lips:
    #             lips = lips_temp[0]
    #     else:
    #         if lips_temp >= lips:
    #             lips = lips_temp

    # g_lips, network_output_range = lipschitz(f_details, box, output_index, activation)
    # if isinstance(g_lips, np.ndarray):
    #     g_lips = g_lips[0]
    #     print('global lips: {}'.format(g_lips))
    lips, network_output_range = lipschitz(f_details, box, output_index, activation)
    if isinstance(lips, np.ndarray):
        lips = lips[0]
    # print('---------------' + filename + '-------------------')
    # print('Lipschitz constant: {}'.format(lips))


    error_bound_lips = lips/2
    range_max = 0.0
    temp = 0
    for j in range(m):
        d_j = d[j]
        temp = temp + 1/d_j
        # lower bound of the j-th component
        alpha_j = box[j][0]
        # upper bound of the j-th component
        beta_j = box[j][1]
        if beta_j - alpha_j > range_max:
            range_max = beta_j - alpha_j
        error_bound_lips = error_bound_lips
    error_bound_lips = error_bound_lips * math.sqrt(temp) * range_max

    x = sp.symbols('x:' + str(f_details.num_of_inputs))
    #b, poly_min, poly_max = nn_poly_approx_bernstein(f, x, d, box, output_index)
    #error_bound_interval = max([poly_min-network_output_range[0][0][0], network_output_range[0][1][0]-poly_max, 0])
    #if error_bound_interval <= np.finfo(np.float64).eps:
    #    error_bound_interval = 0.0

    #print('network_output_range: {}'.format(np.reshape(network_output_range[0],
    #                                                   (1, -1))[0]))
    #print('poly_range: {}'.format([poly_min, poly_max]))
    print('error_bound_lips: {}'.format(error_bound_lips))
    #print('error_bound_interval: {}'.format(error_bound_interval))
    #if error_bound_interval >= error_bound_lips:
    #    flag = '0,'
    #else:
    #    flag = '1,'
    #with open('outputs/times/' + filename + '_count.txt', 'a+') as file:
    #    file.write(flag)
    #with open('outputs/errors/' + filename + '_lip_errors.txt', 'a+') as file:
    #    file.write("{}\n".format(error_bound_lips))
    #with open('outputs/errors/' + filename + '_interval_errors.txt', 'a+') as file:
    #    file.write("{}\n".format(error_bound_interval))
#    return min([error_bound_lips, error_bound_interval])
    return error_bound_lips


##############################################################
def lipschitz(NN_controller, network_input_box, output_index, activation):
    weight_all_layer = NN_controller.weights
    bias_all_layer = NN_controller.bias
    offset = NN_controller.offset
    scale_factor = NN_controller.scale_factor

    layers = len(bias_all_layer)
    lips = 1
    input_range_layer = np.float64(network_input_box)
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
    return lips * scale_factor, (np.array(input_range_layer, dtype=np.float64)-offset) * scale_factor


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
                singular_j = np.array([0.25], dtype=np.float64)
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
                singular_j = np.array([1.0], dtype=np.float64)
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
        b = bias[j][0]
        # compute the minimal input
        res_min = linprog(c, bounds=input_range_layer, options={"disp": False})
        input_j_min = np.array([res_min.fun + b], dtype=np.float64)
        # compute the minimal output
        if activation == 'ReLU':
            if input_j_min < 0:
                output_j_min = np.array([0.0], dtype=np.float64)
            else:
                output_j_min = input_j_min
        if activation == 'sigmoid':
            output_j_min = 1/(1+np.exp(-input_j_min))
        if activation == 'tanh':
            output_j_min = 2/(1+np.exp(-2*input_j_min))-1
        # compute the maximal input
        res_max = linprog(-c, bounds=input_range_layer, options={"disp": False})
        input_j_max = np.array([-res_max.fun + b], dtype=np.float64)
        # compute the maximal output
        if activation == 'ReLU':
            if input_j_max < 0:
                output_j_max = np.array([0.0], dtype=np.float64)
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
