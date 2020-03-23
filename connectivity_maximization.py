#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from socp.rr_socp import ChannelModel
from socp.rr_socp_tests import plot_config, numpy_to_ros
from math import pi
import random
import cvxpy as cp
from scipy.linalg import null_space


def circle_points(rad, num_points):
    angles = np.linspace(0.0, 2.0*pi, num=num_points, endpoint=False)
    pts = np.zeros((num_points,2))
    pts[:,0] = rad*np.cos(angles)
    pts[:,1] = rad*np.sin(angles)
    return pts


class GraphConnectivity:
    def __init__(self, print_values=True, n0=-70.0, n=2.52, l0=-53.0, a=0.2, b=6.0):
        self.cm = ChannelModel(print_values=print_values, n0=n0, n=n, l0=l0, a=a, b=b)

    def compute_connectivity(self, x):
        rate, _ = self.cm.predict(x)
        lap = np.diag(np.sum(rate, axis=1)) - rate
        v, _ = np.linalg.eigh(lap)
        return v[1]


def channel_derivative_quiver():
    cm = ChannelModel()

    xi = np.asarray([0.0, 0.0])
    xj = np.asarray([1.0, 1.0])
    cm = ChannelModel()
    xy = np.zeros((0,2))
    for i in range(3,30,3):
        xy = np.vstack((xy, circle_points(i, 5)))
    uv = np.zeros_like(xy)
    for i in range(uv.shape[0]):
        uv[i,:] = cm.derivative(xy[i,:], np.asarray([0.,0.])).T

    fig, ax = plt.subplots()
    ax.quiver(xy[:,0], xy[:,1], uv[:,0], uv[:,1], scale=4.0)
    ax.axis('equal')
    plt.show()


def connectivity_distance_test():
    task_rad = 20
    comm_agent_count = 3
    lambda2_points = 40

    x_task = circle_points(task_rad, 3)

    gc = GraphConnectivity(print_values=False)
    rad = np.linspace(0.1, 20.0, num=lambda2_points, endpoint=False)
    lambda2 = np.zeros_like(rad)
    for i in range(lambda2_points):
        x_comm = circle_points(rad[i], comm_agent_count)
        lambda2[i] = gc.compute_connectivity(np.vstack((x_task, x_comm)))

    fig, ax = plt.subplots()
    ax.plot(rad, lambda2)
    plt.show()


def derivative_test():

    cm = ChannelModel(print_values=False)

    pts = 100
    xi = np.asarray([0, 0])
    xj = np.zeros((pts,2))
    xj[:,1] = np.linspace(0.01, 30.0, num=pts)
    dist = np.linalg.norm(xj, axis=1)

    rate = np.zeros((pts,))
    for i in range(pts):
        rate[i], _ = cm.predict_link(xi, xj[i,:])

    idx = random.randint(0,pts)
    step = 20
    start_idx = max(idx-20, 0)
    end_idx = min(idx+20, pts)
    Rxixj, _ = cm.predict_link(xi, xj[idx,:])
    dRdxi = cm.derivative(xi, xj[idx,:])
    dRdxj = cm.derivative(xj[idx,:], xi)
    Rtaylor = Rxixj + np.matmul(dRdxi.T, (xi - xi)) \
        + np.matmul(dRdxj.T, (xj[start_idx:end_idx,:] - xj[idx,:]).T)

    fig, ax = plt.subplots()
    ax.plot(dist, rate, 'r', linewidth=2)
    ax.plot(dist[start_idx:end_idx], np.reshape(Rtaylor, (Rtaylor.size,)), 'b', linewidth=2)
    ax.plot(dist[idx], Rxixj, 'bo', markersize=8, fillstyle='none', mew=2)
    plt.show()


# TODO use cp.parameter
def local_controller(gc, x0, comm_agent_idcs, step_size):

    agent_count = x0.shape[0]
    comm_agent_count = len(comm_agent_idcs)
    task_agent_count = agent_count - comm_agent_count

    gamma = cp.Variable((1))
    x = cp.Variable((comm_agent_count,2))
    Aij = cp.Variable((agent_count, agent_count))
    Lij = cp.Variable((agent_count, agent_count))

    # relative closeness

    constraints = [cp.norm(x - x0[comm_agent_idcs], 1) <= step_size * comm_agent_count]

    # linearized rate model

    for i in range(agent_count):
        for j in range(agent_count):
            if i == j:
                constraints += [Aij[i,j] == 0.0]
                continue
            xi = x0[i,:]
            xj = x0[j,:]
            Rxixj, _ = gc.cm.predict_link(xi, xj)
            dRdxi = gc.cm.derivative(xi, xj)
            dRdxj = gc.cm.derivative(xj, xi)
            if i in comm_agent_idcs and j in comm_agent_idcs:
                xi_var = x[comm_agent_idcs.index(i),:]
                xj_var = x[comm_agent_idcs.index(j),:]
                constraints += [Aij[i,j] == Rxixj + dRdxi.T * (xi_var - xi).T \
                                                  + dRdxj.T * (xj_var - xj).T]
            elif i in comm_agent_idcs:
                xi_var = x[comm_agent_idcs.index(i),:]
                constraints += [Aij[i,j] == Rxixj + dRdxi.T * (xi_var - xi).T]
            elif j in comm_agent_idcs:
                xj_var = x[comm_agent_idcs.index(j),:]
                constraints += [Aij[i,j] == Rxixj + dRdxj.T * (xj_var - xj).T]
            else:
                constraints += [Aij[i,j] == Rxixj]

    # graph laplacian

    constraints += [Lij == cp.diag(cp.sum(Aij, axis=1)) - Aij]

    # 2nd smallest eigen value

    P = null_space(np.ones((1, agent_count)))
    constraints += [P.T * Lij * P >> gamma*np.eye(agent_count-1)]

    #
    # solve problem
    #

    prob = cp.Problem(cp.Maximize(gamma), constraints)
    prob.solve()

    if prob.status is not 'optimal':
        print(prob.status)

    x0[comm_agent_idcs] = x.value

    return gc.compute_connectivity(x0)


def algebraic_connectivity_sdp():

    gc = GraphConnectivity(print_values=False)

    x0 = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0], [-2.0, 5.0], [25.0, 10.0]])
    # x0 = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    comm_agent_idcs = [3, 4]
    # x0 = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0], [2.0, 0.0], [4.0, 0.0], [20.0, 10.0]])
    # comm_agent_idcs = [3, 4, 5]
    step_size = 0.2

    fig, ax = plt.subplots()
    plot_config(numpy_to_ros(x0), ax=ax, show=True, pause=1.0,
                title="lambda 2 = {:.3f}".format(gc.compute_connectivity(x0)))

    update = 1.0
    lambda20 = 0.0
    it = 1
    while (update > 1e-10):
        lambda2 = local_controller(gc, x0, comm_agent_idcs, step_size)
        plot_config(numpy_to_ros(x0), ax=ax, clear_axes=True, show=True, pause=0.01,
                    title="it: {:3d}, lambda 2 = {:.3f}".format(it, lambda2))
        update = lambda2 - lambda20
        lambda20 = lambda2
        it += 1

    print("best lambda2 = {:.6f}".format(lambda2))
    plot_config(numpy_to_ros(x0), ax=ax, clear_axes=True, show=True,
                title="total its: {:3d}, lambda 2 = {:.3f}".format(it-1, lambda2))

if __name__ == '__main__':
    algebraic_connectivity_sdp()
