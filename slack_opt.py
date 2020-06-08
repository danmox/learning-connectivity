from bayes_opt import BayesianOptimization
import numpy as np
from socp.rr_socp_tests import plot_config, socp_info
from socp.rr_socp import RobustRoutingSolver
from socp.channel_model import PiecewiseChannel
from socp.msg import Flow
from scipy.spatial import distance_matrix
from network_planner.connectivity_optimization import ConnectivityOpt
import matplotlib.pyplot as plt


def in_proximity(pts, threshold):
    dm = distance_matrix(pts,pts)
    return np.any(dm[~np.eye(dm.shape[0], dtype=bool)] < threshold)


def sample_config(dim, bbx, proximity_threshold):
    """
    dim - number of samples to draw
    bbx - sample bounding box specified as [x_min, x_max, y_min, y_max]
    """
    pts = np.zeros((2,2))
    while in_proximity(pts, proximity_threshold):
        pts = np.random.random((dim,2))*(bbx[1::2] - bbx[0::2]) + bbx[0::2]
    return pts


if __name__ == '__main__':

    # params

    task_count = 4
    comm_count = 3
    bbx = np.asarray([-10, 20, 0, 30])
    proximity_threshold = 1.5
    rate = 0.05
    conf = 0.75
    channel_model = PiecewiseChannel(print_values=False)

    #

    x_task = sample_config(task_count, bbx, proximity_threshold)
    x_task_bbx = np.asarray([min(x_task[:,0]), max(x_task[:,0]),
                             min(x_task[:,1]), max(x_task[:,1])])
    x_comm = sample_config(comm_count, x_task_bbx, proximity_threshold)
    while in_proximity(np.vstack((x_task, x_comm)), proximity_threshold):
        x_comm = sample_config(comm_count, x_task_bbx, proximity_threshold)
    x = np.vstack((x_task, x_comm))

    task_ids = set(range(1, task_count+1))
    ids = set(range(1, task_count+comm_count+1))
    id_to_idx = {id: idx for idx, id in enumerate(ids)}
    flows = [Flow(rate, s, d, Flow.CONFIDENCE, conf) for s in task_ids for d in task_ids if d != s]
    for i, f in enumerate(flows):
        print('flow {:2d}: {} -> {}, r = {:.2f}, q = {:.2f}'.format(i, f.src, f.dest, f.rate, f.qos))

    # initial config

    print('\nINITIAL CONFIGURATION')

    lambda20 = ConnectivityOpt.connectivity(channel_model, x_task, x_comm)

    rrs = RobustRoutingSolver(channel_model)
    slack0, routes0, status = rrs.solve_socp(flows, x, reshape_routes=True)
    if status == 'optimal':
        slack0_str = '{:.4f}'.format(slack0)
        print('slack0 = {}, lambda20 = {:.4f}'.format(slack0_str, lambda20))
        # socp_info(routes0, flows)
    else:
        print('FAILED with status {}'.format(status))
        slack0_str = status
        routes0 = None

    # maximizing connectivity

    print('\nCONNECTIVITY MAXIMIZING CONFIGURATION')

    co = ConnectivityOpt(channel_model, x_task, x_comm)
    lambda21 = co.maximize_connectivity(viz=True)
    x_star1 = co.config

    slack1, routes1, status = rrs.solve_socp(flows, x_star1, reshape_routes=True)
    if status == 'optimal':
        slack1_str = '{:.4f}'.format(slack1)
        print('slack1 = {}, lambda21 = {:.4f}'.format(slack1_str, lambda21))
        # socp_info(routes1, flows)
    else:
        print('FAILED with status {}'.format(status))
        slack1_str = status
        routes1 = None

    # bayseian optimization for mazimal slack

    print('\nSLACK MAXIMIZING CONFIGURATION')

    pbounds = {}
    for i in range(comm_count):
        pbounds['x'+str(i)] = x_task_bbx[0:2]
        pbounds['y'+str(i)] = x_task_bbx[2:]

    def compute_slack(*args, **kwargs):
        x_comm = np.asarray([[kwargs['x'+i], kwargs['y'+i]] for i in map(str, range(comm_count))])
        x_team = np.vstack((x_task, x_comm))
        slack, _, status = rrs.solve_socp(flows, x_team)
        if status == 'optimal':
            return slack
        return 0.0

    x_guess = {}
    for i, idx in enumerate([id_to_idx[j] for j in ids - task_ids]):
        x_guess['x'+str(i)] = x_star1[idx,0]
        x_guess['y'+str(i)] = x_star1[idx,1]

    # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    bopt = BayesianOptimization(f=compute_slack, pbounds=pbounds, verbose=2, random_state=1)
    bopt.probe(params=x_guess, lazy=True)
    bopt.maximize(init_points=5, n_iter=50)
    x_comm = np.asarray([[bopt.max['params']['x'+i],
                          bopt.max['params']['y'+i]] for i in map(str, range(comm_count))])

    lambda22 = ConnectivityOpt.connectivity(channel_model, x_task, x_comm)

    x_star2 = np.vstack((x_task, x_comm))
    slack2, routes2, status = rrs.solve_socp(flows, x_star2, reshape_routes=True)
    if status == 'optimal':
        slack2_str = '{:.4f}'.format(slack1)
        print('slack2 = {}, lambda22 = {:.4f}'.format(slack2_str, lambda22))
        # socp_info(routes2, flows)
    else:
        print('FAILED with status {}'.format(status))
        slack2_str = status
        routes2 = None

    # plots

    fig, axes = plt.subplots(nrows=1, ncols=3)
    plot_config(x, ax=axes[0], show=False, ids=ids, task_ids=task_ids, routes=routes0,
                title='initial: slack = {}, l2 = {:.4f}'.format(slack0_str, lambda20))

    plot_config(x_star1, ax=axes[1], show=False, ids=ids, task_ids=task_ids, routes=routes1,
                title='connectivity: slack = {}, l2 = {:.4f}'.format(slack1_str, lambda21))

    plot_config(x_star2, ax=axes[2], show=False, ids=ids, task_ids=task_ids, routes=routes2,
                title='bayesian: slack = {:.4f}, l2 = {:.4f}'.format(slack2, lambda22))
    plt.show()
