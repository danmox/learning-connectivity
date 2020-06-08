#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from socp.opportunistic_routing import OpportunisticRoutingSolver
from socp.rr_socp_tests import plot_config, numpy_to_ros, socp_info
from socp.msg import Flow
from geometry_msgs.msg import Point


if __name__ == '__main__':

    x = np.asarray([[0., 0.], [15., 0.], [7.5, 3.]])
    flow1 = [Flow(0.2, 1, 0, Flow.OPPORTUNISTIC, 0.0)]
    flow2 = [Flow(0.2, 1, 0, Flow.OPPORTUNISTIC, 0.4)]
    ids = range(0, x.shape[0])
    task_ids = set(ids) - set([2])

    solver = OpportunisticRoutingSolver(print_values=False)
    rate1, var1, routes1, _ = solver.compute_routes(flow1, x, reshape_routes=True)
    socp_info(routes1, flow1, ids=ids)

    rate2, var2, routes2, _ = solver.compute_routes(flow2, x, reshape_routes=True)
    socp_info(routes2, flow2, ids=ids)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plot_config(x, ax=axes[0], show=False, ids=ids, task_ids=task_ids, routes=routes1,
                title='rate = {:.4f}, var = {:.4f}'.format(rate1, var1))

    plot_config(x, ax=axes[1], show=False, ids=ids, task_ids=task_ids, routes=routes2,
                title='rate = {:.4f}, var = {:.4f}'.format(rate2, var2))
    plt.show()
