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

    solver = OpportunisticRoutingSolver(print_values=False)
    max_rate, routes, status = solver.compute_routes(flow1, x, reshape_routes=True)
    print('max_rate = {}'.format(max_rate))
    socp_info(routes, flow1)

    max_rate, routes, status = solver.compute_routes(flow2, x, reshape_routes=True)
    socp_info(routes, flow2)
