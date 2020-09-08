from socp.channel_model import PiecewiseChannel, ChannelModel, LinearChannel
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi
import numpy as np

# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


def circle_points(rad, num_points):
    angles = np.linspace(0.0, 2.0*pi, num=num_points, endpoint=False)
    pts = np.zeros((num_points,2))
    pts[:,0] = rad*np.cos(angles)
    pts[:,1] = rad*np.sin(angles)
    return pts


def derivative_test(cm=PiecewiseChannel()):

    pts = 100
    xi = np.asarray([0, 0])
    xj = np.zeros((pts,2))
    xj[:,1] = np.linspace(0.01, 30.0, num=pts)
    dist = np.linalg.norm(xj, axis=1)

    rate = np.zeros((pts,))
    for i in range(pts):
        rate[i], _ = cm.predict_link(xi, xj[i,:])

    idx = np.random.randint(0,pts)
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


def channel_plot():
    pwcm = PiecewiseChannel(print_values=False)
    lcm = LinearChannel(print_values=False)
    cm = ChannelModel(print_values=False)

    x = np.linspace(0.1, 40.0, num=100)
    pw_rate = np.zeros(x.shape)
    cm_rate = np.zeros(x.shape)
    lm_rate = np.zeros(x.shape)
    xi = np.asarray([0.0, 0.0])
    for i in range(x.shape[0]):
        xj = np.asarray([0.0, x[i]])
        pw_rate[i], _ = pwcm.predict_link(xi, xj)
        lm_rate[i], _ = lcm.predict_link(xi, xj)
        cm_rate[i], _ = cm.predict_link(xi,xj)

    fig, ax = plt.subplots()
    ax.plot(x, cm_rate, 'r', lw=2)
    ax.plot(x, pw_rate, 'b--', lw=2)
    ax.plot(x, lm_rate, 'g-.', lw=2)
    ax.set_xlabel('distance (m)', fontsize='xx-large')
    plt.show()


if __name__ == '__main__':
    # print('running derivative_test()')
    # derivative_test()
    print('running channel_plot()')
    channel_plot()
