# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#
# Visualization
#

import numpy as np
import matplotlib.pyplot as plt

def V(sz):
    return np.eye(sz) - 1/sz*np.ones([1,sz])*np.ones([sz,1])

# requires x = [x1, x2, ... , xN] with xi being a column vector
def D(x):
    if x.shape[0] > x.shape[1]:
        raise Exception('requires: x = [x1,...,xN] with each xi a column vector')
    G = np.matmul(np.transpose(x),x)
    n = G.shape[0]
    dG = np.matrix(np.diag(G))
    return np.matmul(np.transpose(dG),np.ones([1,n])) + np.matmul(np.ones([n,1]),dG) - 2*G

# params
comm_radius = 0.6
client_count = 3
network_count = 1

# randomize node placements
clients = 2*np.random.rand(client_count,2)-1
agents = np.random.rand(network_count,2)-0.5
nodes = np.concatenate((clients,agents))

D(np.transpose(nodes))

# plot graph
agent_idx = client_count + network_count - 1;
for i in range(0,nodes.shape[0]-1):
    plt.plot(nodes[[i,agent_idx],0], nodes[[i,agent_idx],1],
             linestyle='solid',
             color=(0.7, 0.7, 0.7),
             linewidth=1.0)
dots1, = plt.plot(clients[:,0], clients[:,1], 'ro')
dots2, = plt.plot(agents[:,0], agents[:,1], 'bo')
plt.legend((dots1, dots2), ('clients', 'agents'))
plt.show()

#
# Maximization
#