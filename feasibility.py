import matplotlib.pyplot as plt
import numpy as np
from socp.channel_model import PiecewiseChannel, ChannelModel, LinearChannel
from socp.rr_socp_tests import plot_config, numpy_to_ros
from network_planner.connectivity_optimization import ConnectivityOpt
from scipy import spatial
from scipy.sparse.csgraph import minimum_spanning_tree
from math import ceil, floor


def true_idcs(arr):
    return {i for i, el in enumerate(arr) if el}


def connect_graph(points, count, max_range):
    """
    Approximate a steiner tree with bounded edge length and fixed steiner points
    by computing the minimum spanning tree of the given points and placing comm
    agents along edges greater than the comm_range until either all edges are
    less than comm_range or comm_count agents have been placed

    inputs:
      points   - Nx2 array of points to try to connect
      count    - number of additional points that can be added
      max_range - maximum allowable edge length

    outputs:
      new_points - Mx2 positions of the added points
      success    - whether the resulting graph is connected
    """

    d = spatial.distance_matrix(points, points)
    mst = np.asarray(minimum_spanning_tree(d).toarray())

    mst_edges = []
    for i in range(mst.shape[0]):
        for j in true_idcs(mst[i,:] > 0):
            mst_edges.append([i,j,mst[i,j]])
    mst_edges = sorted(mst_edges, key=lambda item: item[2], reverse=True)

    it = 0
    new_points = np.zeros((count,2))
    while it < count and mst_edges[it][2] > max_range:
        xi = points[mst_edges[it][0],:]
        xj = points[mst_edges[it][1],:]
        dist = mst_edges[it][2]

        arrow = (xj - xi) / dist
        for i in range(floor(dist / max_range)):
            new_points[it,:] = xi + arrow * dist * (i+1) / ceil(dist / max_range)
            it += 1
            if it == count:
                break
    new_points = new_points[:it,:]

    config = np.vstack((points, new_points))
    d = spatial.distance_matrix(config, config)
    mst = np.asarray(minimum_spanning_tree(d).toarray())
    if np.max(mst) > max_range:
        success = False
    else:
        success = True

    return new_points, success


def reduce_dispersion(points, count, max_range):
    """
    Add additional points to a set so as to reduce dispersion

    inputs:
      points    - the set of points to reduce dispersion
      count     - the number of points that can be added
      max_range - the maximum allowable distance between points

    outputs:
      new_points - the added points that reduce dispersion
    """

    new_points = np.zeros((0,2))
    for i in range(count):
        vor = spatial.Voronoi(points)
        tri = spatial.Delaunay(points)
        disp_points = vor.vertices[tri.find_simplex(vor.vertices) >= 0,:]
        points_dists = np.linalg.norm(points[:, np.newaxis] - disp_points, axis=2)
        disp_dists = np.amin(points_dists, axis=0)

        candidate_points = disp_dists[disp_dists < max_range]
        if candidate_points.shape[0] == 0:
            break
        pt_idx = np.where(disp_dists == np.max(candidate_points))[0][0]
        new_points = np.vstack((new_points, disp_points[pt_idx]))
        points = np.vstack((points, disp_points[pt_idx]))

    return new_points


def construct_feasible_config(x_task, comm_agents, comm_range):

    x_comm, success = connect_graph(x_task, comm_agents, comm_range)
    if not success:
        return np.zeros((0,2)), False
    elif x_comm.shape[0] != comm_agents:
        config = np.vstack((x_task, x_comm))
        new_points = reduce_dispersion(config, comm_agents - x_comm.shape[0], comm_range)
        x_comm = np.vstack((x_comm, new_points))

    if x_comm.shape[0] != comm_agents:
        return np.zeros((0,2)), False

    return x_comm, True


def feasible_sample(task_agents, comm_agents, comm_range, bbx):
    success = False
    while not success:
        x_task = np.random.random((task_agents,2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
        x_comm, success = construct_feasible_config(x_task, comm_agents, comm_range)
    return x_task, x_comm


def feasibility_test():

    task_agents = 8
    comm_agents = 6
    comm_range = 30  # meters

    # NOTE: generated using the following command (not used here to avoid
    # circular import):
    # bbx = adaptive_bbx(task_agents + comm_agents, comm_range, 0.4)
    bbx = np.asarray([-62.91587036,  62.91587036, -62.91587036,  62.91587036])

    x_task = np.random.random((task_agents,2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
    x_comm, success = construct_feasible_config(x_task, comm_agents, comm_range)
    if not success:
        print(f'failed to find initial feasible configuration')
        return

    plot_config(np.vstack((x_task, x_comm)), task_ids=range(task_agents))


if __name__ == '__main__':
    feasibility_test()
