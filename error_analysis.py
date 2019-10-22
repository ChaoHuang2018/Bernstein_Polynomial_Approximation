from scipy.special import comb
from numpy import linalg as LA
from scipy.optimize import linprog
from polyval import polyval
import tensorflow as tf
import tf_util as U

import numpy as np
import sympy as sp
import itertools


def nn_poly_approx_bernstein(
    f,
    state_vars,
    degree_bound,
    input_box,
    output_index
):
    """
    bernstein polynomial approximation of a given function f
    on a general box space
    f: a function
    state_var: the input variable of f
    d: degree bound vector of bernstein polynomial
    box: box space of state variables [alpha_1, beta_1] times cdots
    times [alpha_m, beta_m]
    """
    input_dim = len(state_vars)
    x = state_vars
    all_comb_lists = degree_comb_lists(degree_bound, input_dim)
    bernstein = 0
    poly_min = np.inf
    poly_max = -np.inf
    # construct bernstein polynomial for recover function + nerual network
    y = sp.symbols('y:'+str(input_dim))
    for cb in all_comb_lists:
        point = []
        for j in range(input_dim):
            k_j = cb[j]
            d_j = degree_bound[j]
            # linear transformation to normalize the box to I=[0,1]^m
            # lower bound of the j-th component
            alpha_j = np.float64(input_box[j][0])
            # upper bound of the j-th component
            beta_j = np.float64(input_box[j][1])
            point.append(
                (beta_j - alpha_j) *
                (cb[j] / degree_bound[j])
                + alpha_j
            )
        monomial = f(np.array(point, dtype=np.float64))[output_index]
        if monomial < poly_min:
            poly_min = monomial
        if monomial > poly_max:
            poly_max = monomial
        for j in range(input_dim):
            y_j = y[j]
            k_j = cb[j]
            d_j = degree_bound[j]
            monomial = (
                monomial * comb(d_j, k_j) *
                (y_j**k_j) * ((1 - y_j)**(d_j - k_j))
            )
        bernstein = bernstein + monomial
    poly_approx = bernstein[0]
    for j in range(input_dim):
        y_j = y[j]
        x_j = x[j]
        alpha_j = np.float64(input_box[j][0])
        beta_j = np.float64(input_box[j][1])
        poly_approx = poly_approx.subs(
            y_j, (x_j - alpha_j) / (beta_j - alpha_j)
        )
    return poly_approx, poly_min[0], poly_max[0]


def nn_poly_approx_bernstein_cuda(f, d, box, output_index):
    m = len(d)
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
    return np.array(all_comb_lists), np.array(coef_list)


def point_shift_all(points, box):
    new_points = np.ones_like(points)
    for idxState in range(points.shape[1]):
        alpha_j = np.float64(box[idxState][0])
        beta_j = np.float64(box[idxState][1])
        new_points[:, idxState] = (
            (points[:, idxState] - alpha_j) /
            (beta_j - alpha_j)
        )
    return new_points


step = -1


def bernstein_error_partition_cuda(
    nn,
    f,
    degree_bound,
    input_box,
    output_index,
    activation,
    filename,
    eps=1e-3
):
    global step
    step += 1

    input_dim = len(degree_bound)
    lips, network_output_range = lipschitz(
        nn,
        input_box,
        output_index,
        activation
    )

    distance_estimate = 0
    for idxState in range(input_dim):
        diff = np.diff(input_box[idxState])[0]
        if diff > distance_estimate:
            distance_estimate = diff

    LD_estimate = lips * distance_estimate * np.sqrt(input_dim)
    num_partition = int(np.ceil(LD_estimate // eps + 1))

    partition = [num_partition] * input_dim
    all_comb_lists = degree_comb_lists(partition, input_dim)

    if isinstance(lips, np.ndarray):
        lips = lips[0]

    print('---------------' + filename + '-------------------')
    print('step: {}'.format(step))
    print('degree bound: {}'.format(degree_bound))
    print('number of partition: {}'.format(num_partition))
    print('Lipschitz constant: {}'.format(lips))

    all_sample_points = np.zeros(
        (len(all_comb_lists), input_dim),
        dtype=np.float64
    )
    all_shift_points = np.zeros(
        (len(all_comb_lists), input_dim),
        dtype=np.float64
    )
    partition_box = np.zeros(input_dim, dtype=np.float64)
    for j in range(input_dim):
        alpha_j = np.float64(input_box[j][0])
        beta_j = np.float64(input_box[j][1])
        partition_box[j] = (beta_j - alpha_j) / num_partition

    all_comb_lists = np.array(all_comb_lists)
    for j in range(input_dim):
        alpha_j = np.float64(input_box[j][0])
        beta_j = np.float64(input_box[j][0])
        all_sample_points[:, idxState] = (
            (beta_j - alpha_j) * (all_comb_lists[:, idxState]/num_partition)
            + alpha_j
        )
        all_shift_points = point_shift_all(all_sample_points, input_box)

    order_list, coeffs_list = nn_poly_approx_bernstein_cuda(
        f,
        degree_bound,
        input_box,
        output_index
    )
    poly = polyval(order_list, degree_bound, coeffs_list, 'test')
    with U.make_session() as sess:
        sess.run(tf.global_variables_initializer())
        poly_results = poly(sess, all_shift_points)
        nn_results = nn(sess, all_sample_points)

    sample_error = np.max(np.absolute(poly_results[:, 0] - nn_results[:, 0]))
    print('sample error: {}'.format(sample_error))
    error = sample_error + lips * LA.norm(partition_box)

    return error


def lipschitz(NN_controller, network_input_box, output_index, activation):
    weight_all_layer = NN_controller.weights
    bias_all_layer = NN_controller.bias
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
        lipschitz_j = lipschitz_layer(
            weight_j,
            bias_j,
            input_range_layer,
            activation_all_layer[j]
        )
        lips = lips * lipschitz_j
        input_range_layer, _ = output_range_layer(
            weight_j,
            bias_j,
            input_range_layer,
            activation_all_layer[j]
        )
    return lips * scale_factor, 0


def lipschitz_layer(weight, bias, input_range_layer, activation):
    neuron_dim = bias.shape[0]
    output_range_box, new_weight = output_range_layer(
        weight,
        bias,
        input_range_layer,
        activation
    )
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
        b = bias[j]
        # compute the minimal input
        res_min = linprog(c, bounds=input_range_layer, options={"disp": False})
        input_j_min = res_min.fun + b
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
        res_max = linprog(
            -c,
            bounds=input_range_layer,
            options={"disp": False}
        )
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
