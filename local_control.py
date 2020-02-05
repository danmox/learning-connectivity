#!/usr/bin/env python

import copy
import numpy as np
from socp.msg import QoS
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
from socp.rr_socp import RobustRoutingSolver
from socp.rr_socp_tests import plot_config, socp_info, numpy_to_ros

import pdb

#
# initialize scenario
#

max_its = 20
sample_count = 100
var = 0.5
x = np.asarray([[ 0.0, 0.0],
                [15.0, 0.0],
                [ 3.0, 0.0]])
task_idcs = np.asarray((0,1))
comm_idcs = np.asarray((2))
qos = QoS()
qos.margin = 0.10
qos.confidence = 0.80
qos.src = 1
qos.dest = [2]
qos_list = [qos]

n = x.shape[0]
k = len(qos_list)

fig, ax = plt.subplots()

#
# local search
#

# initial solution
rrsolver = RobustRoutingSolver(print_values=False)
print_update = True
for it in range(1,max_its):

    slack, routes, status = rrsolver.solve_socp(qos_list, x, reshape_routes=True)
    if print_update:
        print('\niteration %3d' % (it))
        print('slack = %.4f' % (slack))
        socp_info(routes, qos_list, x, solver=rrsolver)
        with np.printoptions(precision=3, suppress=True):
            print(x)
        print_update = False

    # draw samples and plot them over the configuration
    samples = np.random.normal(0.0, var, (comm_idcs.size, 2, sample_count))
    ax.cla()
    ax.plot(samples[:,0,:] + x[comm_idcs,0],
            samples[:,1,:] + x[comm_idcs,1], marker='.', color=(1.0, 0.7, 0.7))
    plot_config(numpy_to_ros(x), ax, show=False)
    plt.pause(0.1)

    for j in range(sample_count):

        # perturb network config (TODO broadcasting doesn't work with +=?)
        dx = np.zeros_like(x)
        dx[comm_idcs,:] = samples[:,:,j]
        x_prime = x + dx

        # find slack of new configuration given current routes
        slack_vec = rrsolver.compute_slack(qos_list, x_prime, routes)
        slack_prime = slack_vec.min()

        if abs(slack_prime - slack) > 1e-7 and slack_prime > slack:
            slack = slack_prime
            x = copy.deepcopy(x_prime)
            print_update = True

print('search complete')
plot_config(numpy_to_ros(x), ax, clear_axes=True)
