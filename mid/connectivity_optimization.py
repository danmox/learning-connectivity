import numpy as np
from math import pi, floor, log10
import random
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from .utils import plot_config


def round_sf(x, significant_figures):
    if x != 0.0:
        return round(x, -int(floor(log10(abs(x))))+significant_figures-1)
    else:
        return 0.0


class ConnectivityOpt:
    def __init__(self, channel_model, x_task, x_comm):
        self.cm = channel_model
        self.x_task = x_task.astype(np.double)
        self.x_comm = x_comm.astype(np.double)
        self.config = np.vstack((self.x_task, self.x_comm))
        self.comm_count = self.x_comm.shape[0]
        self.agent_count = self.config.shape[0]
        self.comm_idcs = range(self.x_task.shape[0], self.agent_count)

    @classmethod
    def connectivity(cls, channel_model, x_task, x_comm):
        config = np.vstack((x_task, x_comm))
        rate, _ = channel_model.predict(config)
        lap = np.diag(np.sum(rate, axis=1)) - rate
        v, _ = np.linalg.eigh(lap)
        return v[1]

    def get_comm_config(self):
        return self.config[self.comm_idcs]

    def update_network_config(self, step_size, verbose=False):

        gamma = cp.Variable((1))
        x = cp.Variable((self.comm_count, 2))
        Aij = cp.Variable((self.agent_count, self.agent_count))
        Lij = cp.Variable((self.agent_count, self.agent_count))

        # relative closeness

        x_dist = x - self.config[self.comm_idcs]
        constraints = [-step_size <= x_dist, x_dist <= step_size]

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
                    constraints += [Aij[i,j] == Rxixj + dRdxi.T @ (xi_var - xi).T \
                                                      + dRdxj.T @ (xj_var - xj).T]
                elif i in self.comm_idcs:
                    xi_var = x[self.comm_idcs.index(i),:]
                    constraints += [Aij[i,j] == Rxixj + dRdxi.T @ (xi_var - xi).T]
                elif j in self.comm_idcs:
                    xj_var = x[self.comm_idcs.index(j),:]
                    constraints += [Aij[i,j] == Rxixj + dRdxj.T @ (xj_var - xj).T]
                else:
                    constraints += [Aij[i,j] == Rxixj]

        # graph laplacian

        constraints += [Lij == cp.diag(cp.sum(Aij, axis=1)) - Aij]

        # 2nd smallest eigen value

        P = null_space(np.ones((1, self.agent_count)))
        constraints += [P.T @ Lij @ P >> gamma*np.eye(self.agent_count-1)]

        #
        # solve problem
        #

        prob = cp.Problem(cp.Maximize(gamma), constraints)
        prob.solve()

        conn_prev = ConnectivityOpt.connectivity(self.cm, self.x_task, self.x_comm)

        if prob.status != 'optimal':
            if verbose: print(f'optimization failed with status {prob.status}')
            return conn_prev, False

        # only update the configuration if a new optimum was found
        #
        # NOTE with agressive step sizes the result of the optimization may be
        # a configuration that has a worse value for connectivity; in most
        # cases the optimization can recover from this but the calling thread
        # should be notified in order to reduce its step size

        conn_new  = ConnectivityOpt.connectivity(self.cm, self.x_task, x.value)

        # NOTE the one case that the optimization cannot recover from is
        # accidentally driving the problem infeasible due to the linearized
        # channel model operating at or near it's cutoff distance; in this
        # case, the configuration should not be updated
        if conn_new < 1e-5:
            if verbose: print(f'optimization close to infeasibility ({conn_new}), updating nothing')
            return conn_prev, False

        success = True if conn_new > conn_prev else False

        # NOTE if the connectivity gets worse but is not zero we still update
        # the configuration to help with fast convergence
        if verbose: print(f'updating comm agent configuration with status {success}')
        self.x_comm = x.value
        self.config[self.comm_idcs] = self.x_comm
        return conn_new, success


    # for use on a static task team only
    def maximize_connectivity(self, init_step_size=0.5, min_step_size=0.01,
                              m_tol=1e-6, h_tol=1e-5, hist=10, max_its=100,
                              viz=False, verbose=False):

        if viz:
            fig, axes = plt.subplots(1,2)
            task_ids = set(range(self.agent_count)) - set(self.comm_idcs)

        # track lambda 2 over time for stopping crit and visualization
        l2_hist = np.zeros((1,))
        l2_hist[0] = ConnectivityOpt.connectivity(self.cm, self.x_task, self.x_comm)

        best_lambda2 = 0.0
        step_size = init_step_size
        for it in range(max_its):
            if verbose: print(f'iteration {it+1}')
            lambda2, success = self.update_network_config(step_size, verbose)

            # the optimization failed, most likely due to an agressive step size
            if not success:
                step_size = max(min_step_size, 0.75*step_size)
                if verbose: print(f'reduced step size to {step_size}')

            # check if change in lambda 2 has "flatlined"
            l2_hist = np.append(l2_hist, [lambda2])
            l2_line = np.polyfit(range(min(l2_hist.shape[0],hist)), l2_hist[-hist:], 1)
            hist_diff = np.max(np.abs(np.diff(l2_hist[-hist:])))

            if viz:
                rates, _ = self.cm.predict(self.config)
                title = f'it = {it}, ss = {round_sf(step_size,2)}'
                plot_config(self.config, ax=axes[0], clear_axes=True, show=False,
                            task_ids=task_ids, rates=rates, title=title)
                axes[1].cla()
                axes[1].plot(np.maximum(np.asarray([-hist, 0]) + l2_hist.shape[0] - 1, 0),
                             l2_line @ np.asarray([[0,min(l2_hist.shape[0],hist)-1],[1,1]]), 'k', lw=2)
                axes[1].plot(l2_hist, 'r', lw=2)
                axes[1].set_title(f'm = {round_sf(abs(l2_line[0]),2)}, diff = {round_sf(hist_diff,2)}')
                plt.tight_layout()
                plt.pause(0.01)

            # stopping criterion: the change in the value of lambda2 has stagnated
            if abs(l2_line[0]) < m_tol and hist_diff < h_tol:
                if verbose: print(f'stopping criterion reached')
                break

        if viz:
            axes[0].set_title(f'total its = {it}, ss = {round_sf(step_size,2)}')
            plt.draw()
            plt.show()

        return lambda2
