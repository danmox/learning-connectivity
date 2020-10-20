import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from socp.channel_model import PiecewisePathLossModel, PathLossModel, LinearModel
from socp.rr_socp_tests import plot_config, numpy_to_ros
from network_planner.connectivity_optimization import ConnectivityOpt
from scipy import spatial
from scipy.sparse.csgraph import minimum_spanning_tree
from math import ceil, floor
import argparse


def adaptive_bbx(agent_count, comm_range=30.0, scale_factor=0.5):
    """
    scale the bounding box within which agent configurations are randomly
    sampled so that the maximum area covered by the agents is a fixed multiple
    of the area of the bounding box
    """
    side_length = np.sqrt(np.pi * agent_count * scale_factor) * comm_range
    return side_length * np.asarray([-1, 1, -1, 1]) / 2.0


def true_idcs(arr):
    """
    return the indices of the elements in an interable that are True
    """
    return {i for i, el in enumerate(arr) if el}


def connect_graph(points, max_dist):
    """
    Approximate a steiner tree with bounded edge length and fixed steiner points
    by computing the minimum spanning tree of the given points and placing comm
    agents along edges greater than the comm_range either all edges are less than
    max_dist

    inputs:
      points   - Nx2 array of points to try to connect
      max_dist - maximum allowable edge length

    outputs:
      new_points - Mx2 positions of the added points
    """

    edm = spatial.distance_matrix(points, points)
    mst = np.asarray(minimum_spanning_tree(edm).toarray())

    mst_edges = []
    for i in range(mst.shape[0]):
        for j in true_idcs(mst[i,:] > 0):
            mst_edges.append([i,j,mst[i,j]])
    mst_edges = sorted(mst_edges, key=lambda item: item[2], reverse=True)

    it = 0
    new_points = np.zeros((0,2))
    while it < len(mst_edges) and not mst_edges[it][2] < max_dist:
        xi = points[mst_edges[it][0],:]
        xj = points[mst_edges[it][1],:]
        dist = mst_edges[it][2]

        arrow = (xj - xi) / dist
        mid_points = np.zeros((floor(dist / max_dist),2))
        for i in range(mid_points.shape[0]):
            mid_points[i,:] = xi + arrow * dist * (i+1) / ceil(dist / max_dist)

        it += 1
        new_points = np.vstack((new_points, mid_points))

    return new_points


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


def min_feasible_sample(task_agents, comm_range, bbx):
    """
    randomly generate a feasible task, network team pair

    randomly sample a task team configuration with a fixed number of agents and
    construct a cooresponding initial feasible network team configuration; if
    the task team configuration already forms a minimum spanning tree then
    disregard it draw a new sample until one is found that requires the support
    of a network team
    """

    success = False
    while not success:
        x_task = np.random.random((task_agents,2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
        x_comm = connect_graph(x_task, comm_range)
        if x_comm.shape[0] != 0:
            break
    return x_task, x_comm


def feasibility_test(args):

    img_bbx = 160/2 * np.asarray([-1, 1, -1, 1])
    comm_range = 30

    task_agents = np.random.randint(3,10) if args.task_agents is None else args.task_agents
    scale_factor = 1.0 if args.scale_factor is None else args.scale_factor
    comm_range = 30

    sample_bbx = adaptive_bbx(task_agents, comm_range, scale_factor)
    print(sample_bbx)
    x_task, x_comm = min_feasible_sample(task_agents, comm_range, sample_bbx)

    plot_config(np.vstack((x_task, x_comm)), task_ids=range(task_agents), bbx=img_bbx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feasibility tests')
    parser.add_argument('--scale-factor', type=float, help='ratio of area covered by agents to image area')
    parser.add_argument('--task-agents', type=int, help='number of task agent locations to sample')
    p = parser.parse_args()

    mpl.rcParams['figure.dpi'] = 150

    feasibility_test(p)
