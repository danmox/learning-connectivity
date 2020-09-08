import matplotlib.pyplot as plt
import numpy as np
from socp.channel_model import PiecewiseChannel, ChannelModel, LinearChannel
from socp.rr_socp_tests import plot_config, numpy_to_ros
from network_planner.connectivity_optimization import ConnectivityOpt
from hdf5_dataset_utils import adaptive_bbx
from feasibility import feasible_sample
from math import pi
import cvxpy as cp
from scipy.linalg import null_space
import time as systime


def circle_points(rad, num_points):
    angles = np.linspace(0.0, 2.0*pi, num=num_points, endpoint=False)
    pts = np.zeros((num_points,2))
    pts[:,0] = rad*np.cos(angles)
    pts[:,1] = rad*np.sin(angles)
    return pts


def connectivity_distance_test():
    task_rad = 20
    comm_agent_count = 3
    lambda2_points = 40

    x_task = circle_points(task_rad, 3)
    cm = PiecewiseChannel(print_values=False)

    rad = np.linspace(0.1, 20.0, num=lambda2_points, endpoint=False)
    lambda2 = np.zeros_like(rad)
    for i in range(lambda2_points):
        x_comm = circle_points(rad[i], comm_agent_count)
        lambda2[i] = ConnectivityOpt.connectivity(cm, x_task, x_comm)

    fig, ax = plt.subplots()
    ax.plot(rad, lambda2)
    plt.show()


def conn_max_test():

    # NOTE requires 1e-9 to separate
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
    # team_size = 8
    # x_task = circle_points(20, team_size)
    # x_comm = circle_points(18, team_size)

    task_agents = 5
    comm_agents = 3
    comm_range = 30  # meters
    bbx = adaptive_bbx(task_agents + comm_agents, comm_range, 0.4)
    x_task, x_comm = feasible_sample(task_agents, comm_agents, comm_range, bbx)

    cm = PiecewiseChannel(print_values=False)
    # cm = ChannelModel(print_values=False)
    # cm = LinearChannel(max_range=comm_range)
    co = ConnectivityOpt(cm, x_task, x_comm)
    co.maximize_connectivity(step_size=1.0, tol=1e-6, viz=True)

    plot_config(x_task)
    plot_config(co.config, task_ids=range(task_agents))


def scale_test():

    def worker(x_task, x_comm, t_avg, t_std, time, its, k):
        co = ConnectivityOpt(x_task, x_comm)
        t0 = systime.time()
        lambda2, its[k], t_avg[k], t_std[k] = co.maximize_connectivity()
        time[k] = systime.time() - t0
        return

    test_trials = 1 # how many times to run each test
    circle_rad = 20
    # agent_count = np.asarray([4, 5, 6, 8, 10, 12, 14])
    agent_count = np.arange(4,20,2)
    time = np.zeros((agent_count.size,2)) # mean, std
    its = np.zeros((agent_count.size,2))  # mean, std
    ittime = np.zeros((agent_count.size,2)) # mean, std

    for i in range(agent_count.size):
        team_size = agent_count[i] / 2
        x_task = circle_points(circle_rad, team_size)
        x_comm = circle_points(0.2, team_size) # np.zeros((team_size, 2)) + np.random.normal(0.0, 0.05, (team_size, 2))

        trial_times = np.zeros(test_trials)
        trial_its = np.zeros(test_trials)
        it_time_avg = np.zeros(test_trials)
        it_time_std = np.zeros(test_trials)
        print("running {:2d} trials for {:2d} agents".format(test_trials, agent_count[i]))
        for j in range(test_trials):
            worker(x_task, x_comm, it_time_avg, it_time_std, trial_times, trial_its, j)

        # co = ConnectivityOpt(x_task, x_comm)
        # plot_config(numpy_to_ros(co.config), show=True)
        # co.maximize_connectivity()
        # plot_config(numpy_to_ros(co.config), show=True)

        print(trial_times)
        print(trial_its)
        time[i,0] = np.mean(trial_times)
        time[i,1] = np.std(trial_times)
        its[i,0] = np.mean(trial_its)
        its[i,1] = np.std(trial_its)
        ittime[i,0] = np.mean(it_time_avg)
        ittime[i,1] = np.mean(it_time_std) # yes, mean

    fig1, ax1 = plt.subplots()

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

    fig1.tight_layout()

    fig2, ax3 = plt.subplots()
    ax3.errorbar(agent_count, ittime[:,0], yerr=ittime[:,1], linewidth=2)
    ax3.set_ylabel('time (s)')
    ax3.set_xlabel('agent count')

    plt.show()


def run_all_tests():
    connectivity_distance_test()
    conn_max_test()
    scale_test()


if __name__ == '__main__':
    #connectivity_distance_test()
    conn_max_test()
