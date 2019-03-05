from scipy.special import comb
from sympy import *

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
    x = sp.symbols(state_vars)
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

    

