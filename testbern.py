#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
import time
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

## test Marta's approach
##NN_controller = nn_controller_details('nn_13_relu', 'ReLU')
##x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
##b, poly_min, poly_max = bp.nn_poly_approx_bernstein(nn_controller('nn_13_relu', 'ReLU'), x, [2,2], [[0.8, 0.9], [0.5, 0.6]], 0)
##print([poly_min, poly_max])
##lips, output_range = bp.lipschitz(NN_controller, [[0.8, 0.9], [0.5, 0.6]], 0, 'ReLU')
##print('our approach to estimate Lipschitz constant: ')
##print(lips)
###print('error bound based on sampling: ')
###t = time.time()
###print(bp.bernstein_error_partition(NN_controller, nn_controller('nn_13_relu', 'ReLU'), [2,2], [[0.8, 0.9], [0.5, 0.6]], 0, 'ReLU', 'nn_13_relu'))
##t1 = time.time()
###elapsed_sampling = t1 - t
###print('Time for sampling based approach:'+str(elapsed_sampling))
##print('error bound based on nested optimization: ')
##print(bp.bernstein_error_nested(NN_controller, nn_controller('nn_13_relu', 'ReLU'), [2,2], [[0.8, 0.9], [0.5, 0.6]], 0, 'ReLU', 'nn_13_relu'))
##t2 = time.time()
##elapsed_nested = t2 - t1
##print('Time for nested optimization based approach:'+str(elapsed_nested))

# test new approach for estimating sigmoid network's output range
# f = nn_controller('nn_17_sigmoid', 'sigmoid')
# NN_controller = nn_controller_details('nn_18_sigmoid', 'sigmoid')
# x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
# output_l, output_u = bp.output_range_MILP(NN_controller, [[0.3, 0.35], [0.45, 0.5], [0.2, 0.25]], 0)


# test error analysis by cuda
NN_controller = nn_controller('nn_13_relu', 'ReLU')
NN_controller_details = nn_controller_details('nn_13_relu', 'ReLU')
d = [2,2]
box = [[0.8, 0.9], [0.5, 0.6]]
output_index = 0
activation = 'ReLU'
filename = 'nn_13_relu'
x = sp.symbols('x:'+ str(NN_controller_details.num_of_inputs))

b, poly_min, poly_max = bp.nn_poly_approx_bernstein(NN_controller, x, d, box, output_index)
print([poly_min, poly_max])
lips, output_range = bp.lipschitz(NN_controller_details, box, output_index, activation)
print('our approach to estimate Lipschitz constant: ')
print(lips)
print('error bound based on sampling: ')
error = bp.bernstein_error_partition_cuda(NN_controller_details, NN_controller, d, box, output_index, activation, filename)