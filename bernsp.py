from scipy.special import comb
from sympy import *
from numpy import linalg as LA
from numpy import pi, tanh, array, dot
from scipy.optimize import linprog
from multiprocessing import Pool
from functools import partial
from operator import itemgetter
from gurobipy import *
import cvxpy as cp

import numpy as np
import sympy as sp
import itertools
import math
import random
import time
import copy


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

def nn_poly_approx_bernstein_cuda(f, state_vars, d, box, output_index):
    m = len(state_vars)
    all_comb_lists = degree_comb_lists(d, m)
    coef_list = []
    for cb in all_comb_lists:
        point = []
        for j in range(m):
            # linear transformation to normalize the box to I=[0,1]^m
            # lower bound of the j-th component
            alpha_j = np.float64(box[j][0])
            # upper bound of the j-th component
            beta_j = np.float64(box[j][1])
            point.append((beta_j-alpha_j)*(cb[j]/d[j])+alpha_j)
        coef = f(np.array(point, dtype=np.float64))[output_index]
        for j in range(m):
            k_j = cb[j]
            d_j = d[j]
            coef = coef*comb(d_j, k_j)
        coef_list.append(coef)
    return all_comb_lists, coef_list

def point_shift(point, box):
    point_new = np.ones_like(point)
    for j in range(point.shape[0]):
        alpha_j = np.float64(box[j][0])
        beta_j = np.float64(box[j][1])
        point_new[j] = (point[j]-alpha_j)/(beta_j-alpha_j)
    return point_new

steps = -1

def bernstein_error_partition(f_details, f, d, box, output_index, activation, filename, eps=1e-2):
    if filename == 'nn_12_relu':
        eps = 1e-2
    elif filename == 'nn_12_sigmoid':
        eps = 1e-2
    elif filename == 'nn_12_tanh':
        eps = 1e-2
    elif filename == 'nn_12_relu_tanh':
        eps = 1e-3
    elif filename == 'nn_13_relu':
        eps = 1e-3
    elif filename == 'nn_13_sigmoid':
        eps = 5e-3
    elif filename == 'nn_13_tanh':
        eps = 1e-2
    elif filename == 'nn_13_relu_tanh':
        eps = 1e-2
    elif filename == 'nn_13_relu_tanh_1':
        eps = 1e-2
    elif filename == 'nn_13_relu_tanh_100':
        eps = 1e-2
    elif filename == 'nn_13_relu_tanh_origin':
        eps = 1e-2
    elif filename == 'nn_14_relu':
        eps = 1e-2
    elif filename == 'nn_14_sigmoid':
        eps = 5e-3
    elif filename == 'nn_14_tanh':
        eps = 1e-2
    elif filename == 'nn_14_relu_sigmoid':
        eps = 5e-3
    elif filename == 'nn_tora_relu_retrained':
        eps = 1e-2
    elif filename == 'nn_tora_tanh':
        eps = 2e-2
    elif filename == 'nn_tora_relu_tanh':
        eps = 1e-2
    elif filename == 'nn_tora_sigmoid':
        eps = 1e-2
    elif filename == 'nn_16_relu':
        eps = 5e-3
    elif filename == 'nn_16_sigmoid':
        eps = 1e-2
    elif filename == 'nn_16_tanh':
        eps = 1e-2
    elif filename == 'nn_16_relu_tanh':
        eps = 1e-2
    elif filename == 'nn_18_relu':
        eps = 4e-3
    elif filename == 'nn_18_relu_tanh':
        eps = 4e-3
    elif filename == 'nn_18_sigmoid':
        eps = 4e-3
    elif filename == 'nn_18_tanh_new':
        eps = 4e-3
    global steps
    steps += 1
    m = len(d)
    lips, network_output_range = lipschitz(f_details, box, output_index, activation)

    distance_estimate = 0
    for j in range(m):
        diff = np.diff(box[j])[0]
        if diff > distance_estimate:
            distance_estimate = diff

    LD_estimate = lips * distance_estimate * np.sqrt(m)
    num_partition = int(np.ceil(LD_estimate // eps + 1))

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
    p = Pool(1000)
    sample_func = partial(sample_error_analysis, f=f, lips=lips, box=box, bern=bern, m=m, num_partition=num_partition, state_vars=state_vars, output_index=output_index, bern_error=bern_error)
    error_list = p.map(sample_func, all_comb_lists)
    error = np.max(error_list)
    # for cb in all_comb_lists:
    #     box_temp = []
    #     for j in range(m):
    #         k_j = cb[j]
    #         alpha_j = np.float64(box[j][0])
    #         beta_j = np.float64(box[j][1])
    #         box_temp.append([(beta_j-alpha_j)*(cb[j]/num_partition)+alpha_j,(beta_j-alpha_j)*((cb[j]+1)/num_partition)+alpha_j])
    #     vertex_index_list = degree_comb_lists([1]*m, m)
    #     for vertex_index in vertex_index_list:
    #         sample_point = np.zeros(m, dtype=np.float64)
    #         poly = bern
    #         distance_m = 0
    #         for j in range(m):
    #             sample_point[j] = box_temp[j][vertex_index[j]]
    #             poly = poly.subs(state_vars[j], sample_point[j])
    #             distance_m += np.diff(box_temp[j])[0]**2
    #         sample_value = f(sample_point)[output_index]
    #         sample_diff = abs(np.float64(poly) - sample_value)[0]
    #         if sample_diff > bern_error and lips != 0.0:
    #             print('---------------- error ------------------')
    #             print('box: {}'.format(box))
    #             print('bern: {}'.format(bern))
    #             print('box temp: {}'.format(box_temp))
    #             print('sample point: {}'.format(sample_point))
    #             print('sample value is {}, poly is {}'.format(sample_value, poly))
    #             raise ValueError('Sample diff {} is smaller than lip error bound {}'.format(sample_diff, bern_error))
    #         # print('sample difference: {}'.format(sample_diff))
    #         # print('LD error: {}'.format(2 * lips * np.sqrt(m) / 2**m * distance_m))
    #         error_temp = lips * np.sqrt(distance_m) + sample_diff
    #         if error_temp >= error:
    #             error = error_temp
       # piece_bern, _, _ = nn_poly_approx_bernstein(f, state_vars, d, box_temp, output_index)
        # error_piece_to_NN = bernstein_error(f_details, f, d, box_temp, output_index, activation, filename)

        # error_piece_to_bern = error_functions(bern, piece_bern, d, box_temp, state_vars)

        # print('piece to bern error: {}'.format(error_piece_to_bern))

        # if error_piece_to_NN + error_piece_to_bern >= piecewise_error:
        #     piecewise_error = error_piece_to_NN + error_piece_to_bern
    # print('LD error: {}'.format(lips * np.sqrt(distance_m)))
    print('sample error: {}'.format(error))
    if error > bern_error and bern_error != 0:
        error = bern_error

    if error < np.finfo(np.float64).eps:
        error = 0.0
    return error


def sample_error_analysis(cb, f, lips, box, bern, m, num_partition, state_vars, output_index, bern_error):
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
        distance_m = 0
        for j in range(m):
            sample_point[j] = box_temp[j][vertex_index[j]]
            poly = poly.subs(state_vars[j], sample_point[j])
            distance_m += np.diff(box_temp[j])[0]**2
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
        error_temp = lips * np.sqrt(distance_m) + sample_diff
    return error_temp


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
# The following code is an improved approach based on Marta's IJCAI18 work

def bernstein_error_nested(f_details, f, d, box, output_index, activation, filename):
    
    m = len(d)
    lips, network_output_range = lipschitz(f_details, box, output_index, activation)
    state_vars = sp.symbols('x:'+ str(m))
    bern, _, _ = nn_poly_approx_bernstein(f, state_vars, d, box, output_index)
    bern_function = lambdify(state_vars, bern)

    # use lambda expression to define the f-bern and bern-f
    f_bern_difference = lambda x: (f(x)[output_index] - bern_function(*x))[0]
    bern_f_difference = lambda x: (- f(x)[output_index] + bern_function(*x))[0]

    # initialize randomly
    comp_index = m
    sample_comp = random.random()
    sample_temp = np.random.rand(m+1)

    f_network = lambda x: f(x)[output_index][0]
    f_network_neg = lambda x: 0-(f(x)[output_index][0])
    t = time.time()
    difference_minimal,_,_,_ = output_min(f_network, lips, box, comp_index, sample_comp, sample_temp)
    t1=time.time()
    print('Minimal of network output: '+str(difference_minimal))
    print('Time for computing minimal of neural network: '+str(t1-t))
    difference_maximal_neg,_,_,_ = output_min(f_network_neg, lips, box, comp_index, sample_comp, sample_temp)
    t2=time.time()
    print('Maximal of network output: '+str(-difference_maximal_neg))
    print('Time for computing maximal of neural network: '+str(t2-t1))

    difference_minimal,_,_,_ = output_min(f_bern_difference, lips, box, comp_index, sample_comp, sample_temp)
    difference_maximal_neg,_,_,_ = output_min(bern_f_difference, lips, box, comp_index, sample_comp, sample_temp)
    difference_maximal = -difference_maximal_neg

    return max(abs(difference_minimal),abs(difference_maximal))
    

def output_min(ff, lips, box, comp_index, sample_comp, sample_temp):
    
    if comp_index == 0:
        # all the components are determined
        sample_temp[comp_index] = sample_comp
        sample_point = sample_temp[0:-1]
        z = ff(sample_point)
        return z,sample_temp,0,0

    else:
        # recursively do the one-dimensional optimization for the comp_index-th component
        comp_index = comp_index-1
        sample_temp[comp_index+1] = sample_comp
        lb = box[comp_index][0]
        ub = box[comp_index][1]
        maxIter = 2000
        bounderror = 1e-5

        x1 = lb
        x2 = ub

        z1,_,_,_ = output_min(ff, lips, box, comp_index, x1, sample_temp)
        z2,_,_,_ = output_min(ff, lips, box, comp_index, x2, sample_temp)
        #print(np.array([[x1,z1],[x2,z2]]))
        xz_sorted = np.array([[x1,z1],[x2,z2]])

        x_next = calculate_x_next(xz_sorted, lips);
        z_next,_,_,_ = output_min(ff, lips, box, comp_index, x_next, sample_temp)
        z_next_1, z_next_2 = calculate_z_next(xz_sorted, lips, np.array([x_next, z_next]))
        z_all = np.array([z_next_1, z_next_2])
        xz_unsorted = np.concatenate((xz_sorted,np.array([[x_next, z_next]])), axis=0)
        xz_sorted = xz_unsorted[xz_unsorted[:,0].argsort()]
        lower_bound_comp = min(z_all)
        upper_bound_comp = np.amin(xz_sorted[:,1])

        i = 3
        while i < maxIter and (upper_bound_comp-lower_bound_comp>bounderror):
            #print('gap between upper and lower bound: '+str(upper_bound_comp-lower_bound_comp))
            z_starIndex = np.argmin(z_all)
            x_next = calculate_x_next(xz_sorted[z_starIndex:z_starIndex+2,:], lips)
            z_next,_,_,_ = output_min(ff, lips, box, comp_index, x_next, sample_temp)
            z_next_1, z_next_2 = calculate_z_next(xz_sorted[z_starIndex:z_starIndex+2,:], lips, np.array([x_next, z_next]))
            z_all = np.concatenate((z_all[:z_starIndex],np.array([z_next_1, z_next_2]),z_all[z_starIndex+1:]), axis=0)
            xz_unsorted = np.concatenate((xz_sorted,np.array([[x_next, z_next]])), axis=0)
            xz_sorted = xz_unsorted[xz_unsorted[:,0].argsort()]
            lower_bound_comp = min(z_all)
            upper_bound_comp = np.amin(xz_sorted[:,1])
            i = i+1
        z = lower_bound_comp
        xz_sorted_temp = xz_unsorted[xz_sorted[:,1].argsort()]
        sample_temp[comp_index] = xz_sorted_temp[0,0]
        return z,sample_temp,lower_bound_comp,upper_bound_comp


def calculate_x_next(xz_sorted, lips):
    x_next = 0.5*(xz_sorted[0,0]+xz_sorted[1,0]) + 0.5*(xz_sorted[0,1]-xz_sorted[1,1])/lips
    return x_next

def calculate_z_next(xz_sorted, lips, xz_next):
    z_next_1 = 0.5*(xz_next[1]+xz_sorted[0,1])-0.5*lips*(xz_next[0]-xz_sorted[0,0])
    z_next_2 = 0.5*(xz_next[1]+xz_sorted[1,1])-0.5*lips*(xz_sorted[1,0]-xz_next[0])
    return z_next_1, z_next_2

##############################################################
# output range analysis by MILP relaxation
def output_range_MILP(NN_controller, network_input_box, output_index):
    weight_all_layer = NN_controller.weights
    bias_all_layer = NN_controller.bias
    offset = NN_controller.offset
    scale_factor = NN_controller.scale_factor
    activation_all_layer = NN_controller.activations


    # initialization of the input range of all the neurons by naive method
    input_range_all = []
    layers = len(bias_all_layer)
    output_range_last_layer = network_input_box
    for j in range(layers):
        if j < layers - 1:
            weight_j = weight_all_layer[j]
        else:
            weight_j = np.reshape(weight_all_layer[j][output_index], (1, -1))
        if j < layers - 1:
            bias_j = bias_all_layer[j]
        else:
            bias_j = np.reshape(bias_all_layer[j][output_index], (1, -1))
        input_range_layer = neuron_range_layer_basic(weight_j, bias_j, output_range_last_layer, activation_all_layer[j])
        #print(input_range_layer[0][1][0])
        #print('range of layer ' + str(j) + ': ' + str(input_range_layer))
        input_range_all.append(input_range_layer)
        output_range_last_layer, _ = output_range_layer(weight_j, bias_j, output_range_last_layer, activation_all_layer[j])
    print("intput range by naive method: " + str([input_range_layer[0][0], input_range_layer[0][1]]))
    print("Output range by naive method: " + str([(output_range_last_layer[0][0]-offset)*scale_factor, (output_range_last_layer[0][1]-offset)*scale_factor]))

    layer_index = 1
    neuron_index = 0
    
    #print('Output range by naive test: ' + str([input_range_all[layer_index][neuron_index]]))
    # compute by milp relaxation
    network_last_input,_ = neuron_input_range(weight_all_layer, bias_all_layer, layers-1, output_index, network_input_box, input_range_all, activation_all_layer)
    #network_last_input,_ = neuron_input_range(weight_all_layer, bias_all_layer, layer_index, neuron_index, network_input_box, input_range_all, activation_all_layer)
    print("Output range by MILP relaxation: " + str([(sigmoid(network_last_input[0])-offset)*scale_factor, (sigmoid(network_last_input[1])-offset)*scale_factor]))

    range_update = copy.deepcopy(input_range_all)
    for j in range(layers):
        for i in range(len(bias_all_layer[j])):
            _, range_update = neuron_input_range(weight_all_layer, bias_all_layer, j, i, network_input_box, range_update, activation_all_layer)
    print(str(range_update[-1]))
    print(str([(sigmoid(range_update[-1][0][0])-offset)*scale_factor, (sigmoid(range_update[-1][0][1])-offset)*scale_factor]))

    return network_last_input[0], network_last_input[1]

        
# Compute the input range for a specific neuron and return the updated input_range_all
# When layer_index = layers, this function outputs the output range of the neural network
def neuron_input_range(weights, bias, layer_index, neuron_index, network_input_box, input_range_all, activation_all_layer):
    weight_all_layer = weights
    bias_all_layer = bias
    layers = len(bias_all_layer)
    width = max([len(b) for b in bias_all_layer])

    # define large positive number M to enable Big M method
    M = 10e4
    # variables in the input layer
    network_in = cp.Variable((len(network_input_box),1))
    # variables in previous layers
    if layer_index >= 1:
        x_in = cp.Variable((width, layer_index))
        x_out = cp.Variable((width, layer_index))
        z = {}
        z[0] = cp.Variable((width, layer_index), integer=True)
        z[1] = cp.Variable((width, layer_index), integer=True)
    # variables for the specific neuron
    x_in_neuron = cp.Variable()

    constraints = []
    # add constraints for the input layer
    if layer_index >= 1:
        constraints += [ 0 <= z[0] ]
        constraints += [ z[0] <= 1]
        constraints += [ 0 <= z[1]]
        constraints += [ z[1] <= 1]
    for i in range(len(network_input_box)):
        constraints += [network_in[i,0] >= network_input_box[i][0]]
        constraints += [network_in[i,0] <= network_input_box[i][1]]

        #constraints += [network_in[i,0] == 0.7]

    if layer_index >= 1:
        #print(x_in[0,0].shape)
        #print(np.array(weight_all_layer[0]).shape)
        #print(network_in.shape)
        #print((np.array(weight_all_layer[0]) @ network_in).shape)
        #print(bias_all_layer[0].shape)
        constraints += [x_in[:,0:1] == np.array(weight_all_layer[0]) @ network_in + bias_all_layer[0]]

    # add constraints for the layers before the neuron
    for j in range(layer_index):
        weight_j = weight_all_layer[j]
        bias_j = bias_all_layer[j]

        # add constraint for linear transformation between layers
        if j+1 <= layer_index-1:
            weight_j_next = weight_all_layer[j+1]
            bias_j_next = bias_all_layer[j+1]
            
            #print(x_in[:,j+1:j+2].shape)
            #print(weight_j_next.shape)
            #print(x_out[:,j:j+1].shape)
            #print(bias_j_next.shape)
            constraints += [x_in[0:len(bias_j_next),j+1:j+2] == weight_j_next @ x_out[0:len(bias_j),j:j+1] + bias_j_next]

        # add constraint for sigmoid function relaxation
        for i in range(weight_j.shape[0]):
            low = input_range_all[j][i][0]
            upp = input_range_all[j][i][1]
            
            # define slack integers
            constraints += [z[0][i,j] + z[1][i,j] == 1]
            # The triangle constraint for 0<=x<=u
            constraints += [-x_in[i,j] <= M * (1-z[0][i,j])]
            constraints += [x_in[i,j] - upp <= M * (1-z[0][i,j])]
            constraints += [x_out[i,j] - sigmoid(0)*(1-sigmoid(0))*x_in[i,j]-sigmoid(0) <= M * (1-z[0][i,j])]
            constraints += [x_out[i,j] - sigmoid(upp)*(1-sigmoid(upp))*(x_in[i,j]-upp) - sigmoid(upp) <= M * (1-z[0][i,j])]
            constraints += [-x_out[i,j] + (sigmoid(upp)-sigmoid(0))/upp*x_in[i,j] + sigmoid(0) <= M * (1-z[0][i,j])]
            # The triangle constraint for l<=x<=0
            constraints += [x_in[i,j] <= M * (1-z[1][i,j])]
            constraints += [-x_in[i,j] + low <= M * (1-z[1][i,j])]
            constraints += [-x_out[i,j] + sigmoid(0)*(1-sigmoid(0))*x_in[i,j] + sigmoid(0) <= M * (1-z[1][i,j])]
            constraints += [-x_out[i,j] + sigmoid(low)*(1-sigmoid(low))*(x_in[i,j]-low) + sigmoid(low) <= M * (1-z[1][i,j])]
            constraints += [x_out[i,j] - (sigmoid(low)-sigmoid(0))/low*x_in[i,j] - sigmoid(0) <= M * (1-z[1][i,j])]

    # add constraint for the last layer and the neuron
    weight_neuron = np.reshape(weight_all_layer[layer_index][neuron_index], (1, -1))
    bias_neuron = np.reshape(bias_all_layer[layer_index][neuron_index], (1, -1))
    #print(x_in_neuron.shape)
    #print(weight_neuron.shape)
    #print(x_out[0:len(bias_all_layer[layer_index-1]),layer_index-1:layer_index].shape)
    #print(bias_neuron.shape)
    if layer_index >= 1:
        constraints += [x_in_neuron == weight_neuron @ x_out[0:len(bias_all_layer[layer_index-1]),layer_index-1:layer_index] + bias_neuron]
    else:
        constraints += [x_in_neuron == weight_neuron @ network_in[0:len(network_input_box),0:1] + bias_neuron]    
  
    # objective: smallest output of [layer_index, neuron_index]
    objective_min = cp.Minimize(x_in_neuron)
    
    prob_min = cp.Problem(objective_min, constraints)
    prob_min.solve(solver=cp.GUROBI)

    if prob_min.status == 'optimal':
        l_neuron = prob_min.value
        #print('lower bound: ' + str(l_neuron))
        #for variable in prob_min.variables():
        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
    else:
        print('prob_min.status: ' + prob_min.status)
        print('Error: No result for lower bound!')

    # objective: largest output of [layer_index, neuron_index]
    objective_max = cp.Maximize(x_in_neuron)
    prob_max = cp.Problem(objective_max, constraints)
    prob_max.solve(solver=cp.GUROBI)

    if prob_max.status == 'optimal':
        u_neuron = prob_max.value
        #print('upper bound: ' + str(u_neuron))
        #for variable in prob_max.variables():
        #    print ('Variable ' + str(variable.name()) + ' value: ' + str(variable.value))
    else:
        print('prob_max.status: ' + prob_max.status)
        print('Error: No result for upper bound!')

    input_range_all[layer_index][neuron_index] = [l_neuron, u_neuron]
    return [l_neuron, u_neuron], input_range_all

        

def neuron_range_layer_basic(weight, bias, output_range_last_layer, activation):
    # solving LPs
    neuron_dim = bias.shape[0]
    input_range_box = []
    for j in range(neuron_dim):
        # c: weight of the j-th dimension
        c = weight[j]
        c = c.transpose()
        #print('c: ' + str(c))
        b = bias[j]
        #print('b: ' + str(b))
        # compute the minimal input
        res_min = linprog(c, bounds=output_range_last_layer, options={"disp": False})
        input_j_min = res_min.fun + b
        # compute the maximal input
        res_max = linprog(-c, bounds=output_range_last_layer, options={"disp": False})
        input_j_max = -res_max.fun + b
        input_range_box.append([input_j_min, input_j_max])
    return input_range_box

## Constraints of MILP relaxation for different layers
    

# define relu activation function and its left/right derivative
def relu(x):
    if x >= 0:
        r = x
    else:
        r = 0
    return r

def relu_de_left(x):
    if x <= 0:
        de_l = 0
    else:
        de_l = 1
    return de_l

def relu_de_right(x):
    if x < 0:
        de_r = 0
    else:
        de_r = 1
    return de_r

# define tanh activation function and its left/right derivative
def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t

def tanh_de_left(x):
    de_l = 1 - (tanh(x))**2
    return de_l

def tanh_de_right(x):
    de_r = tanh_de_left(x)
    return de_r

# define sigmoid activation function and its left/right derivative
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_de_left(x):
    de_l = sigmoid(x)*(1-sigmoid(x))
    return de_l

def sigmoid_de_right(x):
    de_r = sigmoid_de_left(x)
    return de_r
##############################################################
def lipschitz(NN_controller, network_input_box, output_index, activation):
    weight_all_layer = NN_controller.weights
    bias_all_layer = NN_controller.bias
    offset = NN_controller.offset
    scale_factor = NN_controller.scale_factor

    activation_all_layer = NN_controller.activations

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
        lipschitz_j = lipschitz_layer(weight_j, bias_j, input_range_layer, activation_all_layer[j])
        lips = lips * lipschitz_j
        input_range_layer, _ = output_range_layer(weight_j, bias_j, input_range_layer, activation_all_layer[j])
    return lips* scale_factor, 0

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



