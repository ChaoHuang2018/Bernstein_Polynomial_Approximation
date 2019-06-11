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

f = nn_controller('nn_16_sigmoid', 'sigmoid')
f_details = nn_controller_details('nn_16_sigmoid', 'sigmoid')
x = sp.symbols('x:'+ str(3))
#box =  [[-0.3800015123696811, -0.3509407750439703], [0.2896937563173715, 0.312567309199469], [0.6715839217369094, 0.6976966788396981]]
box = [[-0.3800037597641404, -0.3509380119715122], [0.2896917855817571, 0.3125687618960456], [0.6715746351007925, 0.6977053148497618]]
#box = [[0,1],[0,1]]
b, poly_min, poly_max =  bp.nn_poly_approx_bernstein(f, x, [3,3,3], box, 0)
lips_error = bp.bernstein_error_partition(f_details, f, [3,3,3], box, 0, 'sigmoid', 'nn_16_sigmoid', 0.00001)
print('Lips error: ' + str(lips_error))
#point =  np.array([ 1.05018855, -0.20917678])
#point = np.array([ (box[0][1]-box[0][0])*2/3+box[0][0], (box[1][1]-box[1][0])*2/3+box[1][0]])
point = np.array([box[0][0], box[1][0], box[2][0]])
#print(point)
#print(f(point)[0])
a = b.subs(x[0], box[0][0])
c=a.subs(x[1], box[1][0])
c=c.subs(x[2], box[2][0])
print(c)
print(f(point)[0])
print("difference: {}".format(abs(c-f(point)[0])))
