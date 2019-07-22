import bernsp as bp
import numpy as np
import sympy as sp
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = nn_controller('nn_12_relu', 'ReLU')
f_details = nn_controller_details('nn_12_relu', 'ReLU')
x_sym = sp.symbols('x:'+str(2))

box = [[0.1, 0.9], [0.1, 0.9]]

sample_range = np.array(box)

b, _, _ = bp.nn_poly_approx_bernstein(f, x_sym, [1, 1], box, 0)

dense = 0.05
x = np.arange(sample_range[0][0], sample_range[0][1], dense)
y = np.arange(sample_range[1][0], sample_range[1][1], dense)
X, Y = np.meshgrid(x, y)
inputs = np.array([X.flatten(), Y.flatten()]).T

num_data = inputs.shape[0]
x_data = np.zeros(num_data)
y_data = np.zeros(num_data)
f_data = np.zeros(num_data)
p_data = np.zeros(num_data)


for i in range(num_data):
    input = inputs[i]
    x_data[i] = input[0]
    y_data[i] = input[1]
    f_data[i] = f(input)[0]
    poly = b
    for j in range(2):
        poly = poly.subs(x_sym[j], input[j])
    p_data[i] = poly
    # print('poly: {}, nn: {}'.format(p_data[i], f_data[i]))

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.cla()
ax.scatter(x_data, y_data, f_data, label='nn')
ax.scatter(x_data, y_data, p_data, label='bernstein_poly', alpha=0.5)
ax.legend()

plt.savefig('plot_bernsp.pdf')

input = np.zeros(2)
for i in range(2):
    for j in range(2):
        input[0] = box[0][i]
        input[1] = box[1][j]
        f_res = f(input)[0]
        poly_res = b
        for indx in range(2):
            poly_res = poly_res.subs(x_sym[indx], input[indx])
            print('diff: {}'.format(f_res-poly_res))
