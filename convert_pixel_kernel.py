#!/usr/bin/env python3

import json
import numpy as np
from scipy.stats import norm
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


# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


def human_readable_duration(dur):
    t_str = []
    for unit, name in zip((86400., 3600., 60., 1.), ('d','h','m','s')):
        if dur / unit > 1.:
            t_str.append(f'{int(dur / unit)}{name}')
            dur -= int(dur / unit) * unit
    return ' '.join(t_str)


def kernelized_config_img(task_config, params):

    img = np.zeros(params['img_size'])
    for agent in task_config:
        dist = np.linalg.norm(params['xy'] - agent, axis=2)
        mask = dist < 3.0*params['kernel_std']
        img[mask] = np.maximum(img[mask], norm.pdf(dist[mask], scale=params['kernel_std']))
    img *= 255.0 / norm.pdf(0, scale=params['kernel_std']) # normalize image to [0.0, 255.0]
    return img


def convert_hdf5_data(in_hdf5_file, out_hdf5_file, mode, params):
    """
    initialize hdf5 datastructure

    file structure is:
    hdf5 file
     - ...
     - mode
       - task_config
       - init_img
       - comm_config
       - final_img
       - connectivity
    """

    out_hdf5_grp = out_hdf5_file.create_group(mode)

    sample_count = in_hdf5_file[mode]['comm_config'].shape[0]
    out_hdf5_grp.create_dataset('task_config', (sample_count, params['task_agents'], 2), np.float64)
    out_hdf5_grp.create_dataset('init_img', [sample_count,] + params['img_size'], np.uint8)
    out_hdf5_grp.create_dataset('comm_config', (sample_count, params['comm_agents'], 2), np.float64)
    out_hdf5_grp.create_dataset('final_img', [sample_count,] + params['img_size'], np.uint8)
    out_hdf5_grp.create_dataset('connectivity', (sample_count,), np.float64)

    # image generation loop

    bbx = np.asarray(params['bbx'])
    t0 = time.time()

    for i in range(sample_count):

        task_config = in_hdf5_file[mode]['task_config'][i,...]
        out_hdf5_grp['task_config'][i,...] = task_config

        # initial configuration of task agents as image
        out_hdf5_grp['init_img'][i,...] = kernelized_config_img(task_config, params)

        # configuration of network agents as numpy array
        out_hdf5_grp['comm_config'][i,...] = in_hdf5_file[mode]['comm_config'][i,...]

        # configuration of entire team as image
        final_config = np.vstack((task_config, in_hdf5_file[mode]['comm_config'][i,...]))
        out_hdf5_grp['final_img'][i,...] = kernelized_config_img(final_config, params)

        # connectivity
        out_hdf5_grp['connectivity'][i] = in_hdf5_file[mode]['connectivity'][i]

        duration_str = human_readable_duration(time.time()-t0)
        print(f'converted sample {i+1} of {sample_count}, elapsed time: {duration_str}\r', end="")


if __name__ == '__main__':

    # params

    kernel_std = 1.0  # meters

    # parse input

    parser = argparse.ArgumentParser(description='generate dataset for learning connectivity')
    parser.add_argument('--dataset', type=str, help=f'Dataset to convert', required=True)
    parser.add_argument('--display', action='store_true', help='display a sample of the data after generation')
    p = parser.parse_args()

    # init

    in_dataset = Path(p.dataset)
    if not in_dataset.exists():
        print(f'the dataset {in_dataset} was not found')
        exit(-1)

    in_params = Path(p.dataset).with_name(in_dataset.stem + '.json')
    if not in_params.exists():
        print(f'the parameter file {str(in_params)} was not found')
        exit(-1)
    with open(in_params, 'r') as f:
        params = json.load(f)
    params['kernel_std'] = kernel_std

    out_dataset = in_dataset.with_name(in_dataset.stem + '_kernel.hdf5')

    in_hdf5_file = h5py.File(in_dataset, mode='r')
    out_hdf5_file = h5py.File(out_dataset, mode='w')

    out_params = in_params.with_name(in_params.stem + '_kernel.json')
    with open(out_params, 'w') as f:
        json.dump(params, f, indent=4, separators=(',', ': '))

    img_width = params['img_size'][0]
    i, j = np.meshgrid(np.arange(img_width), np.arange(img_width), indexing='ij')
    ij = np.stack((i, j), axis=2)
    xy = params['meters_per_pixel'] * (ij + 0.5)
    params['ij'] = ij
    params['xy'] = xy

    # conversion

    for mode in ('train','test'):

        print(f'converting {mode}ing samples')
        convert_hdf5_data(in_hdf5_file, out_hdf5_file, mode, params)

    # view a selection of the generated data
    if p.display:
        num_viz_samples = 3
        sample_counts = [out_hdf5_file[m]['connectivity'].shape[0] for m in ('train','test')]
        sample_idcs = [np.random.randint(0, max_idx, size=(num_viz_samples,)) for max_idx in sample_counts]
        for idcs, mode in zip(sample_idcs, ('train','test')):
            for i, idx in enumerate(idcs):
                plt.subplot(num_viz_samples, 2, i*2+1)
                plt.imshow(out_hdf5_file[mode]['init_img'][idx,...])
                plt.subplot(num_viz_samples, 2, (i+1)*2)
                plt.imshow(out_hdf5_file[mode]['final_img'][idx,...])
            plt.show()

    print(f'saved data to: {out_hdf5_file.filename}')
    out_hdf5_file.close()
    in_hdf5_file.close()
