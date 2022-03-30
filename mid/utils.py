import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_config(config, ax=None, pause=None, clear_axes=False, show=True,
                title=None, ids=None, task_ids=None, routes=None, rates=None,
                bbx=None):
    """Plot the 2D spatial configuration of the network.

    Input:
      config : a Nx2 list of x,y agent positions
      Optional Args:
        ax : axes to plot on
        pause : avoids blocking by continuing after a short pause
        clear_axes : clear ax before plotting
        show : call plt.show()
        title : string of text to set as title of figure
        ids : as list of ids to use for agent labels
        task_ids : ids of task agents
        routes : draw lines denoting route usage between nodes
        rates : draw lines denoting rates between agents
        bbx : bounding box to use for figure area

    """
    x = config[:,0]
    y = config[:,1]

    if ax is None:
        fig, ax = plt.subplots()

    if clear_axes:
        ax.cla()

    if ids is None:
        ids = range(len(x))
    elif type(ids) is not list:
        ids = list(ids)

    if task_ids is None:
        id_mask = np.asarray(len(ids)*[True], dtype=bool)
    else:
        id_mask = np.asarray([True if id in task_ids else False for id in ids], dtype=bool)

    # draw routes between each agent

    ax.axis('scaled')
    if bbx is None:
        bbx = np.asarray([min(x), max(x), min(y), max(y)])
        window_scale = np.max(bbx[1::2] - bbx[0::2])
        ax.axis(bbx + np.asarray([-1, 1, -1, 1])*0.1*window_scale)
    else:
        window_scale = np.max(bbx[1::2] - bbx[0::2])
        ax.axis(bbx)

    if routes is not None:

        # form dict of all route lines to be plotted later
        cumulative_routes = np.sum(routes, 2)
        lines = []
        for i, j in [(i,j) for i in range(len(x)) for j in range(i+1,len(x))]:
            Pi = np.asarray([x[i], y[i]])
            Pj = np.asarray([x[j], y[j]])
            Aij = cumulative_routes[i,j]
            Aji = cumulative_routes[j,i]

            # ensures the arrows are oriented correctly
            if Pj[0] < Pi[0] or Pj[1] < Pi[1]:
                Pi, Pj = Pj, Pi
                Aij, Aji = Aji, Aij

            a1 = np.arctan2(Pj[1]-Pi[1], Pj[0]-Pi[0])
            a2 = np.arctan2(Pi[1]-Pj[1], Pi[0]-Pj[0])

            # line segment endpoints
            ds = np.pi / 16.0
            scale = 0.03 * window_scale
            l1 = np.zeros((2,2))
            l2 = np.zeros((2,2))
            l1[0,:] = Pi + scale*np.asarray([np.cos(a1+ds), np.sin(a1+ds)])
            l1[1,:] = Pj + scale*np.asarray([np.cos(a2-ds), np.sin(a2-ds)])
            l2[0,:] = Pi + scale*np.asarray([np.cos(a1-ds), np.sin(a1-ds)])
            l2[1,:] = Pj + scale*np.asarray([np.cos(a2+ds), np.sin(a2+ds)])

            # arrowhead endpoint
            ds = np.pi / 8.0
            scale = 0.04 * window_scale
            h1 = l1[0,:] + scale*np.asarray([np.cos(a1+ds), np.sin(a1+ds)])
            h2 = l2[1,:] + scale*np.asarray([np.cos(a2+ds), np.sin(a2+ds)])

            if Aji > 0.01:
                lines.append({'rate': Aji, 'line_x': l1[:,0], 'line_y': l1[:,1],
                              'arrow_x': [l1[0,0], h1[0]], 'arrow_y': [l1[0,1], h1[1]]})
            if Aij > 0.01:
                lines.append({'rate': Aij, 'line_x': l2[:,0], 'line_y': l2[:,1],
                              'arrow_x': [l2[1,0], h2[0]], 'arrow_y': [l2[1,1], h2[1]]})

        # plot lines by line weight: faintest on the bottom, boldest on top
        lw = 2
        cmap = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0, clip=True), cmap='YlOrBr')
        lines.sort(key=lambda line_dict: line_dict['rate'])
        for d in lines:
            ax.plot(d['line_x'], d['line_y'], lw=lw, c=cmap.to_rgba(d['rate']))
            ax.plot(d['arrow_x'], d['arrow_y'], lw=lw, c=cmap.to_rgba(d['rate']))

    # plot the rate graph (if no routes are provided)

    if routes is None and rates is not None:
        cmap = cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0, clip=True), cmap='plasma')
        for i, j in [(i,j) for i in range(len(x)) for j in range(i+1,len(x))]:
            if rates[i,j] == 0.0:
                continue

            Pi = np.asarray([x[i], y[i]])
            Pj = np.asarray([x[j], y[j]])

            a1 = np.arctan2(Pj[1]-Pi[1], Pj[0]-Pi[0])
            a2 = np.arctan2(Pi[1]-Pj[1], Pi[0]-Pj[0])

            # line segment endpoints
            scale = 0.04 * window_scale
            line = np.zeros((2,2))
            line[0,:] = Pi + scale*np.asarray([np.cos(a1), np.sin(a1)])
            line[1,:] = Pj + scale*np.asarray([np.cos(a2), np.sin(a2)])

            ax.plot(line[:,0], line[:,1], lw=2, c=cmap.to_rgba(rates[i,j]))

        # plt.colorbar(cmap, ax=ax)

    # plot agent positions as circles

    if len(x) != len(id_mask):
        import pdb;pdb.set_trace()
    ax.plot(x[id_mask], y[id_mask], 'ro', ms=16, fillstyle='none',
            markeredgewidth=2, label='task')
    ax.plot(x[~id_mask], y[~id_mask], 'bo', ms=16, fillstyle='none',
            markeredgewidth=2, label='network')
    ax.legend(markerscale=0.6)

    # add agent ID in middle of circle

    for i in range(len(x)):
        color= 'r' if id_mask[i] == True else 'b'
        ax.annotate(str(ids[i]), (x[i], y[i]-0.05), color=color,
                    horizontalalignment='center', verticalalignment='center')

    if title is not None:
        ax.set_title(title)

    if show:
        if pause is None:
            plt.show()
        else:
            plt.pause(pause)
