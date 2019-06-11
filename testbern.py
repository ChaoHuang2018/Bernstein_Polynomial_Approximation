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

f = nn_controller('nn_13_relu', 'ReLU')
f_details = nn_controller_details('nn_13_relu', 'ReLU')
x = sp.symbols('x:'+ str(2))
box =  [[0.9290615235056694, 1.064320037786938], [-0.2634684012809252, -0.142217119554879]]
#box = [[0,1],[0,1]]
b, poly_min, poly_max =  bp.nn_poly_approx_bernstein(f, x, [3,3], box, 0)
lips_error = bp.bernstein_error(f_details, f, [3,3], box, 0, 'ReLU', 'nn_13_relu')
print('Lips error: ' + str(lips_error))
point =  np.array([ 1.05018855, -0.20917678])
#point = np.array([ (box[0][1]-box[0][0])*2/3+box[0][0], (box[1][1]-box[1][0])*2/3+box[1][0]])
print(point)
print(f(point)[0])
a = b.subs(x[0],1.05018855)
c=a.subs(x[1],-0.20917678)
#a=b.subs(x[0],(box[0][1]-box[0][0])*2/3+box[0][0])
#c=a.subs(x[1],(box[1][1]-box[1][0])*2/3+box[1][0])
print(c)
print(abs(c-f(point)[0]))
