#from .NN_Tracking.code.neuralnetwork import NN
import bernsp as bp
import numpy as np
import sympy as sp
from network_controller import dubins_car_nn_controller, dubins_car_nn_controller_details
from numpy import pi, tanh, array, dot

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


weight_all_layer, bias_all_layer = dubins_car_nn_controller_details()
lips = bp.lipschitz(weight_all_layer, bias_all_layer, [[ 2.853499451232261e+00 , 2.937847072787262e+00 ],[ -5.917093026681963e-01 , -5.777043015696386e-01 ]], 'tanh')
print('basic estimation of Lipschitz constant: ')
print(1.5391113341478*pi)
print('our approach to estimate Lipschitz constant: ')
print(lips*pi)
