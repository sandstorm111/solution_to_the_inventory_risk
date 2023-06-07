from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.linalg import expm
import math

#https://arxiv.org/pdf/1105.3115.pdf

## BASIC VARIABLES
max_inventory_Q = 30
min_inventory_Q = -max_inventory_Q
iterations_reference = 2 * max_inventory_Q + 1
sigma = 0.3
A = 0.9
kappa = 0.3
gamma =  0.01
T = 600
dt = 1
u = 1.0

## ALPHA, Nill
alpha = kappa / 2 * gamma * sigma ** 2
beta = kappa * u
nill  = A * (1 + gamma / kappa) ** - (1 + kappa / gamma)

assert len(list(range(min_inventory_Q, max_inventory_Q + 1, 1))) == iterations_reference

matrix_M = []
first_row = []
last_row = []
list_middle_rows = []
for i in range(min_inventory_Q, max_inventory_Q + 1, 1):

    # Populating first row
    if i == min_inventory_Q: 
        first_row.append(alpha * max_inventory_Q ** 2 - beta * max_inventory_Q)
        first_row.append(-nill)
        first_row.extend([0 for j in range(iterations_reference - 2)])

    # Populating last row
    elif i == max_inventory_Q:
        last_row.extend([0 for j in range(iterations_reference - 2)])
        last_row.append(-nill)
        last_row.append(alpha * max_inventory_Q ** 2 - beta * max_inventory_Q)

    # Populating middle rows
    else:
        middle_row = []
        if i < 0:
            middle_row.extend([-nill, alpha * ( max_inventory_Q - (max_inventory_Q - abs(i)) ) ** 2 - beta * ( max_inventory_Q - (max_inventory_Q - abs(i)) ), -nill])
            middle_row.extend([0 for j in range(iterations_reference - 3)])

            if i > -max_inventory_Q + 1:
                for j in range(0, max_inventory_Q + i - 1, 1):
                    middle_row.insert(0,middle_row.pop())

        if i == 0:
            middle_row = [0 for j in range(iterations_reference)]
            middle_row[max_inventory_Q + 1] = -nill
            middle_row[max_inventory_Q - 1] = -nill
            assert len(middle_row) == iterations_reference

        if i > 0:
            middle_row.extend([0 for j in range(iterations_reference - 3)])
            middle_row.extend([-nill, alpha * ( max_inventory_Q - (max_inventory_Q - abs(i)) ) ** 2 - beta * ( max_inventory_Q - (max_inventory_Q - abs(i)) ), -nill])

            for j in range(0,  max_inventory_Q - i - 1, 1):
                middle_row.append(middle_row.pop(0))

        assert len(middle_row) == iterations_reference
        list_middle_rows.append(middle_row)

matrix_M.append(first_row)
matrix_M.extend(list_middle_rows)
matrix_M.append(last_row)

assert len(first_row) == iterations_reference
assert len(last_row) == iterations_reference

# Converting to numpy array
matrix_M = np.matrix(matrix_M)

# Simulating negative inventory
list_evolution_bid = []
list_evolution_ask = []
for t in range(0, T  , 1):
    dict_bid = {}
    dict_ask = {}
    list_bid_at_t = []
    container = -matrix_M * (T - t)

    v_of_t = expm(container)
    v_of_t  = np.sum(np.multiply(v_of_t, np.ones(v_of_t.shape[0])), axis=1)

    for q in range(0, iterations_reference -1, 1):
        bid = (1 / kappa) * np.log(v_of_t[q] / v_of_t[q + 1]) + (1 / gamma) * np.log(1 + gamma / kappa)
        ask = (1 / kappa) * np.log(v_of_t[q] / v_of_t[q - 1]) + (1 / gamma) * np.log(1 + gamma / kappa)
        dict_ask[q - max_inventory_Q] = ask
        dict_bid[q - max_inventory_Q] = bid

    dict_bid[max_inventory_Q] = None
    dict_ask[min_inventory_Q] = None
    dict_ask[max_inventory_Q] = dict_bid[min_inventory_Q]

    list_evolution_bid.append(list(dict_bid.values())[1:-2])

# Plotting 
X = np.arange(1, T+1, 1)
Y = np.arange(min_inventory_Q+1, max_inventory_Q-1 , 1)
X, Y = np.meshgrid(X, Y)

Z = np.array(list_evolution_bid).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.invert_xaxis()

plt.show()
