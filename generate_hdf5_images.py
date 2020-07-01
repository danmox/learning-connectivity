#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.misc
from pathlib import Path
import shutil
import os
import inspect
import time
import datetime
import h5py
import argparse

from network_planner.connectivity_optimization import ConnectivityOpt
from socp.channel_model import PiecewiseChannel


# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


def pos_to_subs(res, pts):
    """
    assume origin is at (0,0) and x,y res is equal
    """
    return np.floor(pts / res).astype(int)


def human_readable_duration(dur):
    t_str = []
    for unit, name in zip((86400., 3600., 60., 1.), ('d','h','m','s')):
        if dur / unit > 1.:
            t_str.append(f'{int(dur / unit)}{name}')
            dur -= int(dur / unit) * unit
    return ' '.join(t_str)


def generate_hdf5_data(hdf5_file, mode, sample_count, params):

    # initialize hdf5 datastructure
    #
    # file structure is:
    # hdf5 file
    #  - ...
    #  - mode
    #    - task_config
    #    - init_img
    #    - comm_config
    #    - final_img
    #    - connectivity

    hdf5_grp = hdf5_file.create_group(mode)

    hdf5_grp.create_dataset('task_config', (sample_count, params['task_agents'], 2), np.float64)
    hdf5_grp.create_dataset('init_img', (sample_count,) + params['img_size'], np.uint8)
    hdf5_grp.create_dataset('comm_config', (sample_count, params['comm_agents'], 2), np.float64)
    hdf5_grp.create_dataset('final_img', (sample_count,) + params['img_size'], np.uint8)
    hdf5_grp.create_dataset('connectivity', (sample_count,), np.float64)

    # image generation loop

    bbx = params['bbx']
    t0 = time.time()
    comm_idcs = np.arange(params['comm_agents']) + params['task_agents']

    for i in range(sample_count):

        # initial configuration of task agents as numpy array
        task_config = np.random.random((params['task_agents'],2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
        hdf5_grp['task_config'][i,...] = task_config

        # initial configuration of task agents as binary image
        init_img = np.zeros(params['img_size'], dtype=np.uint8)
        subs = pos_to_subs(params['meters_per_pixel'], task_config)
        init_img[subs[:,0], subs[:,1]] = 1
        hdf5_grp['init_img'][i,...] = init_img

        # configuration of network agents as numpy array
        comm_config = np.random.random((params['comm_agents'],2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]
        conn_opt = ConnectivityOpt(params['channel_model'], task_config, comm_config)
        conn_opt.maximize_connectivity()
        comm_config = conn_opt.config[comm_idcs,:]
        hdf5_grp['comm_config'][i,...] = comm_config

        # configuration of entire team as binary image
        final_img = np.zeros(params['img_size'], dtype=np.uint8)
        subs = pos_to_subs(params['meters_per_pixel'], np.vstack((task_config, comm_config)))
        final_img[subs[:,0], subs[:,1]] = 1
        hdf5_grp['final_img'][i,...] = final_img

        # connectivity
        l2 = ConnectivityOpt.connectivity(params['channel_model'], task_config, comm_config)
        hdf5_grp['connectivity'][i] = l2

        duration_str = human_readable_duration(time.time()-t0)
        print(f'saved sample {i+1} of {sample_count}, elapsed time: {duration_str}\r', end="")


if __name__ == '__main__':

    # params

    task_agents = 4
    comm_agents = 3
    samples = 10
    train_percent = 0.85
    space_side_length = 30  # length of a side of the image in meters
    img_res = 128           # pixels per side of a square image
    datadir = Path(__file__).resolve().parent / 'data'

    # parse input

    parser = argparse.ArgumentParser(description='generate dataset for learning connectivity')
    parser.add_argument('--samples', default=10, type=int, help=f'Number of samples to generate. Default is {samples}')
    parser.add_argument('--display', action='store_true', help='display a sample of the data after generation')
    p = parser.parse_args()

    # init

    params = {}
    params['task_agents'] = task_agents
    params['comm_agents'] = comm_agents
    params['img_size'] = (img_res, img_res)
    params['bbx'] = np.asarray([0, space_side_length, 0, space_side_length])
    params['channel_model'] = PiecewiseChannel(print_values=False)
    params['meters_per_pixel'] = space_side_length / img_res
    print(f"using {img_res}x{img_res} images with {params['meters_per_pixel']} meters/pixel")

    # generate random training data

    hdf5_file = h5py.File(datadir / 'connectivity_from_imgs.hdf5', mode='w')

    sample_counts = np.round(p.samples * (np.asarray([0,1]) + train_percent*np.asarray([1,-1]))).astype(int)
    for count, mode in zip(sample_counts, ('train','test')):
        print(f'generating {count} samples for {mode}ing')

        t0 = time.time()
        generate_hdf5_data(hdf5_file, mode, count, params)
        duration_str = human_readable_duration(time.time()-t0)
        print(f'generated {count} samples for {mode}ing in {duration_sr}')

        if p.display:
            plt.plot(hdf5_file[mode]['connectivity'], 'r.', ms=3)
            plt.show()

    # view a selection of the generated data
    if p.display:
        num_viz_samples = 3
        sample_idcs = [np.random.randint(0, max_idx, size=(num_viz_samples,)) for max_idx in sample_counts]
        for idcs, mode in zip(sample_idcs, ('train','test')):
            for i, idx in zip(range(num_viz_samples), idcs):
                plt.subplot(num_viz_samples, 2, i*2+1)
                plt.imshow(hdf5_file[mode]['init_img'][idx,...])
                plt.subplot(num_viz_samples, 2, (i+1)*2)
                plt.imshow(hdf5_file[mode]['final_img'][idx,...])
            plt.show()

    print(f'saved data to: {hdf5_file.filename}')
    hdf5_file.close()
