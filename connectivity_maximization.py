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


class ConnectivityOpt:
    def __init__(self, print_values=True, n0=-70.0, n=2.52, l0=-53.0, a=0.2, b=6.0,
                 x_task=None, x_comm=None):
        self.cm = ChannelModel(print_values=print_values, n0=n0, n=n, l0=l0, a=a, b=b)
        self.x_task = x_task
        self.x_comm = x_comm
        if x_task is not None and x_comm is not None:
            self.config = np.vstack((self.x_task, self.x_comm))
            self.agent_count = self.config.shape[0]
            self.comm_count = self.x_comm.shape[0]
            self.comm_idcs = range(self.x_task.shape[0], self.agent_count)

    def connectivity(self, x):
        rate, _ = self.cm.predict(x)
        lap = np.diag(np.sum(rate, axis=1)) - rate
        v, _ = np.linalg.eigh(lap)
        return v[1]

    # TODO use cp.parameter?
    def update_network_config(self, step_size):

        gamma = cp.Variable((1))
        x = cp.Variable((self.comm_count, 2))
        Aij = cp.Variable((self.agent_count, self.agent_count))
        Lij = cp.Variable((self.agent_count, self.agent_count))

        # relative closeness

        constraints = [cp.norm(x - self.config[self.comm_idcs], 1) <= step_size * self.comm_count]

        # linearized rate model

        for i in range(self.agent_count):
            for j in range(self.agent_count):
                if i == j:
                    constraints += [Aij[i,j] == 0.0]
                    continue
                xi = self.config[i,:]
                xj = self.config[j,:]
                Rxixj, _ = self.cm.predict_link(xi, xj)
                dRdxi = self.cm.derivative(xi, xj)
                dRdxj = self.cm.derivative(xj, xi)
                if i in self.comm_idcs and j in self.comm_idcs:
                    xi_var = x[self.comm_idcs.index(i),:]
                    xj_var = x[self.comm_idcs.index(j),:]
                    constraints += [Aij[i,j] == Rxixj + dRdxi.T * (xi_var - xi).T \
                                                      + dRdxj.T * (xj_var - xj).T]
                elif i in self.comm_idcs:
                    xi_var = x[self.comm_idcs.index(i),:]
                    constraints += [Aij[i,j] == Rxixj + dRdxi.T * (xi_var - xi).T]
                elif j in self.comm_idcs:
                    xj_var = x[self.comm_idcs.index(j),:]
                    constraints += [Aij[i,j] == Rxixj + dRdxj.T * (xj_var - xj).T]
                else:
                    constraints += [Aij[i,j] == Rxixj]

        # graph laplacian

        constraints += [Lij == cp.diag(cp.sum(Aij, axis=1)) - Aij]

        # 2nd smallest eigen value

        P = null_space(np.ones((1, self.agent_count)))
        constraints += [P.T * Lij * P >> gamma*np.eye(self.agent_count-1)]

        #
        # solve problem
        #

        prob = cp.Problem(cp.Maximize(gamma), constraints)
        prob.solve()

        if prob.status is not 'optimal':
            print(prob.status)

        self.x_comm = x.value
        self.config[self.comm_idcs] = self.x_comm
        return self.connectivity(self.config)


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

    co = ConnectivityOpt(print_values=False)
    rad = np.linspace(0.1, 20.0, num=lambda2_points, endpoint=False)
    lambda2 = np.zeros_like(rad)
    for i in range(lambda2_points):
        x_comm = circle_points(rad[i], comm_agent_count)
        lambda2[i] = co.connectivity(np.vstack((x_task, x_comm)))

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


def algebraic_connectivity_sdp():

    x_task = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0]])
    x_comm = np.asarray([[-2.0, 5.0], [25.0, 10.0]])
    # x_comm = np.asarray([[2.0, 0.0], [4.0, 0.0]])
    # x_comm = np.asarray([[2.0, 0.0], [4.0, 0.0], [20.0, 10.0]])
    step_size = 0.2

    co = ConnectivityOpt(print_values=False, x_task=x_task, x_comm=x_comm)

    fig, ax = plt.subplots()
    plot_config(numpy_to_ros(np.vstack((x_task, x_comm))), ax=ax, show=True, pause=1.0,
                title="lambda 2 = {:.3f}".format(co.connectivity(np.vstack((x_task, x_comm)))))

    update = 1.0
    lambda20 = 0.0
    it = 1
    while (update > 1e-10):
        lambda2 = co.update_network_config(step_size)
        plot_config(numpy_to_ros(co.config), ax=ax, clear_axes=True, show=True, pause=0.01,
                    title="it: {:3d}, lambda 2 = {:.3f}".format(it, lambda2))
        update = lambda2 - lambda20
        lambda20 = lambda2
        it += 1

    print("best lambda2 = {:.6f}".format(lambda2))
    plot_config(numpy_to_ros(co.config), ax=ax, clear_axes=True, show=True,
                title="total its: {:3d}, lambda 2 = {:.3f}".format(it-1, lambda2))

if __name__ == '__main__':
    algebraic_connectivity_sdp()
