#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from socp.rr_socp import ChannelModel
from socp.rr_socp_tests import plot_config, numpy_to_ros
from math import pi
import random
import cvxpy as cp
from scipy.linalg import null_space
import time as systime


def circle_points(rad, num_points):
    angles = np.linspace(0.0, 2.0*pi, num=num_points, endpoint=False)
    pts = np.zeros((num_points,2))
    pts[:,0] = rad*np.cos(angles)
    pts[:,1] = rad*np.sin(angles)
    return pts


class ConnectivityOpt:
    def __init__(self, x_task=None, x_comm=None,
                 print_values=False, n0=-70.0, n=2.52, l0=-53.0, a=0.2, b=6.0):
        self.cm = ChannelModel(print_values=print_values, n0=n0, n=n, l0=l0, a=a, b=b)
        self.x_task = x_task
        self.x_comm = x_comm
        if x_task is not None and x_comm is not None:
            self.config = np.vstack((self.x_task, self.x_comm))
            self.agent_count = self.config.shape[0]
            self.comm_count = self.x_comm.shape[0]
            self.comm_idcs = range(self.x_task.shape[0], self.agent_count)

    def connectivity(self):
        rate, _ = self.cm.predict(self.config)
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
        return self.connectivity()

    def maximize_connectivity(self, step_size=0.2, tol=1e-10, max_its=1000):

        update = 1.0
        lambda2_prev = 0.0
        it = 0
        while update > tol and it < 1000:
            lambda2 = self.update_network_config(step_size)
            update = lambda2 - lambda2_prev
            lambda2_prev = lambda2
            it += 1

        return lambda2, it


def channel_derivative_quiver():
    xi = np.asarray([0.0, 0.0])
    xj = np.asarray([1.0, 1.0])
    cm = ChannelModel(print_values=False)
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

    rad = np.linspace(0.1, 20.0, num=lambda2_points, endpoint=False)
    lambda2 = np.zeros_like(rad)
    for i in range(lambda2_points):
        x_comm = circle_points(rad[i], comm_agent_count)
        co = ConnectivityOpt(x_task=x_task, x_comm=x_comm)
        lambda2[i] = co.connectivity()

    fig, ax = plt.subplots()
    ax.plot(rad, lambda2)
    plt.show()


def derivative_test():

    cm = ChannelModel()

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


def acsdp_circle_test():

    # NOTE requires 1e-10 to separate
    # x_task = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0]])
    # x_comm = np.asarray([[-2.0, 5.0], [25.0, 10.0]])

    # NOTE position wiggle at end up meets convergence criterion
    # x_task = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0]])
    # x_comm = np.asarray([[2.0, 0.0], [4.0, 0.0]])

    # NOTE can start in a cluster if not at a "local minimum"
    # x_task = np.asarray([[0.0, 0.0], [10.0, 20.0], [20.0, 0.0]])
    # x_comm = np.zeros((3,2))

    # NOTE very slow convergence: lambda2 effectively flatlines after ~40-60
    # iterations but the positions appreciably change until the end
    # x_task = circle_points(20, 8)
    # x_comm = np.zeros((8, 2)) + np.random.normal(0.0, 0.01, (8, 2))

    # NOTE same convergence behavior as above; however, despite starting from a
    # different initial config (close to the task agents) the final
    # configuration is often the same
    x_task = circle_points(20, 8)
    x_comm = circle_points(14, 8) + np.random.normal(0.0, 0.01, (8,2))

    step_size = 0.2
    tol = 1e-10

    co = ConnectivityOpt(x_task=x_task, x_comm=x_comm)

    fig, axes = plt.subplots(1,2)

    update = 1.0
    lambda20 = 0.0
    it = 1
    lambda2_hist = np.zeros((1,))
    lambda2_hist[0] = co.connectivity()
    while (update > tol):
        lambda2 = co.update_network_config(step_size)
        lambda2_hist = np.append(lambda2_hist, np.asarray([lambda2]))
        plot_config(numpy_to_ros(co.config), ax=axes[0], clear_axes=True, show=False,
                    title="it: {:3d}, lambda 2 = {:.3f}".format(it, lambda2))
        axes[1].cla()
        axes[1].plot(range(0,it+1), lambda2_hist, 'r', linewidth=2)
        axes[1].set_title("update = {:.4e}".format(lambda2-lambda20))
        plt.tight_layout()
        plt.pause(0.01)
        update = lambda2 - lambda20
        lambda20 = lambda2
        it += 1

    print("best lambda2 = {:.6f}".format(lambda2))
    plot_config(numpy_to_ros(co.config), ax=axes[0], clear_axes=True, show=False,
                title="total its: {:3d}, lambda 2 = {:.3f}".format(it-1, lambda2))
    axes[1].cla()
    axes[1].plot(range(0,it), lambda2_hist, 'r', linewidth=2)
    axes[1].set_title("update = {:.4e}".format(update))
    plt.tight_layout()
    plt.show()


def scale_test():

    test_trials = 10 # how many times to run each test
    circle_rad = 20
    agent_count = np.asarray([4, 6, 8, 12, 16, 20])
    time = np.zeros((agent_count.size,2)) # mean, std
    its = np.zeros((agent_count.size,2)) # mean, std

    for i in range(agent_count.size):
        team_size = agent_count[i] / 2
        x_task = circle_points(circle_rad, team_size)
        x_comm = np.zeros((team_size, 2)) + np.random.normal(0.0, 0.01, (team_size, 2))

        trial_times = np.zeros(test_trials)
        trial_its = np.zeros(test_trials)
        for j in range(test_trials):
            co = ConnectivityOpt(x_task, x_comm)
            t0 = systime.time()
            lambda2, trial_its[j]  = co.maximize_connectivity()
            trial_times[j] = systime.time() - t0
            if j == 0:
                plot_config(numpy_to_ros(co.config), show=True)

        print(trial_times)
        print(trial_its)
        time[i,0] = np.mean(trial_times)
        time[i,1] = np.std(trial_times)
        its[i,0] = np.mean(trial_its)
        its[i,1] = np.std(trial_its)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('agent count')
    ax1.set_ylabel('time (s)', color=color)
    ax1.errorbar(agent_count, time[:,0], yerr=time[:,1], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('iterations', color=color)
    ax2.errorbar(agent_count, its[:,0], yerr=its[:,1], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def run_all_tests():
    channel_derivative_quiver()
    connectivity_distance_test()
    derivative_test()
    acsdp_circle_test()
    scale_test()


if __name__ == '__main__':
    acsdp_circle_test()
