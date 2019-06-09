#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
#import controller_approximation_lib as cal

#a = bp.degree_comb_lists([2,3,31],3)
#print(a)

#error = bp.bernstein_error(1,[2,3,31])
#print(error)


#x1, x2 = sp.symbols('x1 x2')
#x = [x1, x2]
#b = bp.nn_poly_approx_bernstein(bp.test_f, x, [2,2])
#print(bp.p2c(b))


#x1, x2 = sp.symbols('x1 x2')
#x = [x1, x2]
#b = bp.nn_poly_approx_bernstein(bp.test_f, x, [2,2], [[1,31],[2,31]])
#print(bp.p2c(b))


#error = bp.bernstein_error(1,[2,3,31], [[1,31],[2,31],[3,8]])
#print(error)


#x=['d_err','t_err']
#b = bp.nn_poly_approx_bernstein(dubins_car_nn_controller(), x, [2,2], [[1,31],[2,31]])
#print(bp.p2c(b))

NN_controller = nn_controller_details('nn_12_relu', 'ReLU')
x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
b, poly_min, poly_max = bp.nn_poly_approx_bernstein(nn_controller('nn_12_relu', 'ReLU'), x, [1, 1],
                                                    [[0.3, 0.31], [0.3, 0.31]], 0)
print([poly_min, poly_max])
print('lip_error_bound: {}'.format(bp.bernstein_error(NN_controller, nn_controller('nn_12_relu', 'ReLU'), [1, 1],
                         [[0.3, 0.31], [0.3, 0.31]], 0, 'ReLU', 'nn_12_relu')))
print('piece_error: {}'.format(bp.bernstein_error_partition(NN_controller, nn_controller('nn_12_relu', 'ReLU'), [1, 1],
                         [[0.3, 0.31], [0.3, 0.31]], 0, 'ReLU', 'nn_12_relu')))
