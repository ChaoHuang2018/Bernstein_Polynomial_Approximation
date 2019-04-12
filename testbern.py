#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
#import controller_approximation_lib as cal

#a = bp.degree_comb_lists([2,3,4],3)
#print(a)

#error = bp.bernstein_error(1,[2,3,4])
#print(error)


#x1, x2 = sp.symbols('x1 x2')
#x = [x1, x2]
#b = bp.nn_poly_approx_bernstein(bp.test_f, x, [2,2])
#print(bp.p2c(b))


#x1, x2 = sp.symbols('x1 x2')
#x = [x1, x2]
#b = bp.nn_poly_approx_bernstein(bp.test_f, x, [2,2], [[1,4],[2,4]])
#print(bp.p2c(b))


#error = bp.bernstein_error(1,[2,3,4], [[1,4],[2,4],[3,8]])
#print(error)


#x=['d_err','t_err']
#b = bp.nn_poly_approx_bernstein(dubins_car_nn_controller(), x, [2,2], [[1,4],[2,4]])
#print(bp.p2c(b))

NN_controller = nn_controller_details('nn_LLM', 'ReLU')
x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
b, poly_min, poly_max = bp.nn_poly_approx_bernstein(nn_controller('nn_LLM', 'ReLU'), x, [1, 1, 1, 1, 1, 1, 1],
                                                    [[0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51]], 0)
print([poly_min, poly_max])
lips, output_range = bp.lipschitz(NN_controller, [[0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51]], 0, 'ReLU')
print('our approach to estimate Lipschitz constant: ')
print(lips)
print('error bound based on Lipschitz constant: ')
print(bp.bernstein_error(NN_controller, nn_controller('nn_LLM', 'ReLU'), [1, 1, 1, 1, 1, 1, 1],
                         [[0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51], [0.5, 0.51]], 0, 'ReLU'))
print('output range of neural network: ')
print(output_range)
print('output range of poly approximation: ')
print([poly_min, poly_max])
