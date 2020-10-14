#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.misc
from scipy.stats import norm
from pathlib import Path
import shutil
import os
import inspect
import time
import datetime
import h5py
import json
import argparse
from multiprocessing import Queue, Process, cpu_count
import shutil
from math import ceil
import torch
from torch.utils.data import Dataset

from network_planner.connectivity_optimization import ConnectivityOpt as ConnOpt
from socp.channel_model import PiecewisePathLossModel, PathLossModel
from feasibility import adaptive_bbx, min_feasible_sample


class ConnectivityDataset(Dataset):
    """connectivity dataset"""

    def __init__(self, file_path, train=True):
        """assumes file_path exists"""
        self.file_path = file_path
        self.mode = 'train' if train else 'test'
        self.dataset = None

        with h5py.File(file_path, 'r') as f:
            self.dataset_len = f[self.mode]['connectivity'].shape[0]

    def __getitem__(self, idx):
        # in order to use the multiprocessing capabilities provided by pytorch,
        # the hdf5 file must be loaded after __init__ since opened hdf5 files
        # are not pickleable:
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.mode]

        # rescale: [0,255] -> [0,1] and reshape: (X,X) -> (1, X, X)
        x = np.expand_dims(self.dataset['task_img'][idx,...] / 255.0, axis=0)
        y = np.expand_dims(self.dataset['comm_img'][idx,...] / 255.0, axis=0)

        # convert to float tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return (x, y)

    def __len__(self):
        return self.dataset_len


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


def console_width_str(msg):
    col, _ = shutil.get_terminal_size((80,20))
    return msg + (col - len(msg))*' '


def kernelized_config_img(config, params):
    img = np.zeros(params['img_size'])
    for agent in config:
        dist = np.linalg.norm(params['xy'] - agent, axis=2)
        mask = dist < 3.0*params['kernel_std']
        img[mask] = np.maximum(img[mask], norm.pdf(dist[mask], scale=params['kernel_std']))
    img *= 255.0 / norm.pdf(0, scale=params['kernel_std']) # normalize image to [0.0, 255.0]
    return np.clip(img, 0, 255)


def write_hdf5_image_data(params, filename, queue):
    '''
    helper function for writing hdf5 image data in a multiprocessing scenario

    inputs:
      params   - dict of parameters used to generate training data
      filename - name of the hdf5 database in which to store training samples
      queue    - multiprocessing queue in which samples to be written to the
                 database are placed by generating processes

    file structure is:
      hdf5 file
       - ...
       - mode
         - task_config
         - task_img
         - comm_config
         - comm_img
         - connectivity
    '''

    # initialize hdf5 database structure

    hdf5_file = h5py.File(filename, mode='w')
    for mode in ('train', 'test'):
        grp = hdf5_file.create_group(mode)

        sc = params['sample_count'][mode]
        grp.create_dataset('task_config', (sc, params['task_agents'], 2), np.float64)
        grp.create_dataset('task_img', (sc,) + params['img_size'], np.uint8)
        grp.create_dataset('comm_config', (sc, params['max_comm_agents'], 2), np.float64)
        grp.create_dataset('comm_img', (sc,) + params['img_size'], np.uint8)
        grp.create_dataset('connectivity', (sc,), np.float64)

    # monitor queue for incoming samples

    t0 = time.time()
    it_dict = {'test': 0, 'train': 0}
    total_samples = sum([params['sample_count'][mode] for mode in ('train', 'test')])
    total_saved = 0
    for data in iter(queue.get, None):
        mode = data['mode']
        it = it_dict[mode]

        for f in ('task_config', 'task_img', 'comm_config', 'comm_img', 'connectivity'):
            hdf5_file[mode][f][it,...] = data[f]

        it_dict[mode] += 1
        total_saved += 1
        duration_str = human_readable_duration(time.time()-t0)
        msg = console_width_str(f'saved sample {total_saved} of {total_samples}, elapsed time: {duration_str}')
        print('\r' + msg + '\r', end="")

    print(console_width_str(f'generated {total_samples} samples in {duration_str}'))
    print(f'saved data to: {hdf5_file.filename}')
    hdf5_file.close()


def generate_hdf5_image_data(params, sample_queue, writer_queue):
    '''
    helper function for generating hdf5 training samples in a multiprocessing
    scenario

    inputs:
      params       - dict of parameters used to generate training data
      sample_queue - the queue of seed data to generate samples from
      writer_queue - the queue of data to be written to the hdf5 database
    '''

    for d in iter(sample_queue.get, None):
        pad = params['max_comm_agents'] - d['comm_config'].shape[0]
        conn_opt = ConnOpt(params['channel_model'], d['task_config'], d['comm_config'])

        out_dict = {}
        out_dict['mode'] = d['mode']
        out_dict['task_config'] = d['task_config']
        out_dict['task_img'] = kernelized_config_img(d['task_config'], params)
        out_dict['connectivity'] = conn_opt.maximize_connectivity()
        out_dict['comm_config'] = np.vstack((conn_opt.get_comm_config(), np.empty((pad,2)) * np.nan))
        out_dict['comm_img'] = kernelized_config_img(conn_opt.get_comm_config(), params)

        # put sample dict in writer queue to be written to the hdf5 database
        writer_queue.put(out_dict)


def generate_hdf5_dataset(task_agents, samples, jobs):

    # generation params
    max_task_agents = 15    # the maximum number of task agents to design the sample image for
    comm_range = 30         # maximum range of communication hardware
    img_res = 128           # pixels per side of a square image
    train_percent = 0.85    # samples total samples will be divided into training and testing sets
    area_scale_factor = 0.5 # ratio of max area covered by N agents vs area of image for bbx

    kernel_std = img_res*0.05 # standard deviation of gaussian kernel marking node positions
    space_side_length = 2.0*ceil(adaptive_bbx(max_task_agents, comm_range, area_scale_factor).max()+kernel_std)
    sample_counts = np.round(samples*(np.asarray([0,1]) + train_percent*np.asarray([1,-1]))).astype(int)
    sample_bbx = adaptive_bbx(task_agents, comm_range, area_scale_factor) # area within which to draw sample

    if task_agents > max_task_agents:
        print(f'too many task agents ({task_agents}), parameters tuned for {max_task_agents}')
        return

    # params to save to config file
    params = {}
    params['task_agents'] = task_agents
    params['img_size'] = (img_res, img_res)
    params['area_scale_factor'] = area_scale_factor
    params['comm_range'] = comm_range
    params['meters_per_pixel'] = space_side_length / img_res
    params['kernel_std'] = kernel_std

    # generate descriptive filename
    # NOTE there is a risk of overwriting a database if this script is run more
    # than once in a second
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = Path(__file__).resolve().parent / 'data' / \
        f"connectivity_{samples}s_{task_agents}t_{timestamp}.hdf5"

    # save param file
    param_file_name = filename.with_suffix('.json')
    with open(param_file_name, 'w') as f:
        json.dump(params, f, indent=4, separators=(',', ': '))

    # these params don't need to be saved to disk
    params['sample_count'] = {mode: int(count) for count, mode in zip(sample_counts, ('train', 'test'))}
    params['channel_model'] = PiecewisePathLossModel(print_values=False)
    ij = np.stack(np.meshgrid(np.arange(img_res), np.arange(img_res), indexing='ij'), axis=2)
    params['xy'] = params['meters_per_pixel'] * (ij + 0.5) - space_side_length/2.0
    params['max_comm_agents'] = 3 * ceil((sample_bbx[1] - sample_bbx[0]) / comm_range)

    # print dataset info to console
    print(f"using {img_res}x{img_res} images with {params['meters_per_pixel']} meters/pixel"
          f" and {task_agents} task agents")

    # configure multiprocessing and generate training data
    #
    # NOTE samples are generated by a pool of worker processes feeding off the
    # sample_queue; these worker processes in turn push generated samples onto
    # the write_queue which is served by a single processes that handles all
    # hdf5 database operations

    num_processes = cpu_count() if jobs is None else jobs
    sample_queue = Queue(maxsize=num_processes*2) # see NOTE below on queue size
    write_queue = Queue()

    # initialize writer process
    hdf5_proc = Process(target=write_hdf5_image_data, args=(params, filename, write_queue))
    hdf5_proc.start()

    # initialize data generation processes
    worker_processes = []
    for i in range(num_processes):
        p = Process(target=generate_hdf5_image_data, args=(params, sample_queue, write_queue))
        worker_processes.append(p)
        p.start()

    # load sample_queue with seed data used to generate the training samples
    #
    # NOTE calls to sample_queue.put block when sample_queue is full; this is
    # the desired behavior since we may be generating tens of thousands of
    # samples and might not want to load them into memory all at the same time
    print(f'generating {samples} samples using {num_processes} processes')
    img_bbx = (space_side_length/2.0 - kernel_std) * np.asarray([-1,1,-1,1])
    for count, mode in zip(sample_counts, ('train','test')):
        it = 0
        while it < count:
            x_task, x_comm = min_feasible_sample(task_agents, comm_range, sample_bbx)
            if x_comm.shape[0] > params['max_comm_agents']: # NOTE fairly certain this is impossible
                print(f'too many comm agents: {x_comm.tolist()}')
                continue

            # randomly shift the configuration so that all parts of the image
            # are used for all team sizes
            x = np.vstack((x_task, x_comm))
            x_bbx = np.asarray([x[:,0].min(), x[:,0].max(), x[:,1].min(), x[:,1].max()])
            shift_bbx = img_bbx - x_bbx
            shift = np.random.random((1,2)) * (shift_bbx[1::2] - shift_bbx[0::2]) + shift_bbx[0::2]

            sample_queue.put({'mode': mode, 'task_config': x_task+shift, 'comm_config': x_comm+shift})
            it += 1

    # each worker process exits once it receives a None
    for i in range(num_processes):
        sample_queue.put(None)
    for proc in worker_processes:
        proc.join()

    # finally, the writer process also exits once it receives a None
    write_queue.put(None)
    hdf5_proc.join()


def view_hdf5_dataset(dataset_file, samples):

    dataset = Path(dataset_file)
    if not dataset.exists():
        print(f'the dataset {dataset} was not found')
        return

    params_file = dataset.with_suffix('.json')
    if not params_file.exists():
        print(f'the parameter file {str(params_file)} was not found')
        return
    with open(params_file, 'r') as f:
        params = json.load(f)

    hdf5_file = h5py.File(dataset, mode='r')

    sample_idcs = []
    for count in [hdf5_file[m]['task_img'].shape[0] for m in ('train','test')]:
        sample_idcs.append(np.random.choice(count, (min(count, samples),), False))

    bbx = params['img_size'][0] * params['meters_per_pixel'] / 2.0 * np.asarray([-1,1,-1,1])
    for idcs, mode in zip(sample_idcs, ('train','test')):
        idcs.sort()
        print(f"plotting {len(idcs)} {mode}ing samples: {', '.join(map(str, idcs))}")
        for i, idx in enumerate(idcs):
            task_config = hdf5_file[mode]['task_config'][idx,...]
            comm_config = hdf5_file[mode]['comm_config'][idx,...]

            # task agent configuration
            ax = plt.subplot(2,2,1)
            ax.plot(task_config[:,1], task_config[:,0], 'g.', ms=4)
            ax.axis('scaled')
            ax.axis(bbx)
            ax.invert_yaxis()
            plt.subplot(2,2,2)
            plt.imshow(hdf5_file[mode]['task_img'][idx,...])

            # network agent configuration
            ax = plt.subplot(2,2,3)
            ax.plot(task_config[:,1], task_config[:,0], 'g.', ms=4)
            ax.plot(comm_config[:,1], comm_config[:,0], 'r.', ms=4)
            ax.axis('scaled')
            ax.axis(bbx)
            ax.invert_yaxis()
            plt.subplot(2,2,4)
            plt.imshow(hdf5_file[mode]['comm_img'][idx,...])

            plt.suptitle(f'{mode}ing sample {i+1}/{len(idcs)} with index {idx}', fontsize=14)
            plt.show()

    hdf5_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='utilities for hdf5 datasets for learning connectivity')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # generate subparser
    gen_parser = subparsers.add_parser('generate', help='generate connectivity dataset')
    gen_parser.add_argument('samples', type=int, help='number of samples to generate')
    gen_parser.add_argument('task_count', type=int, help='number of task agents')
    gen_parser.add_argument('--jobs', '-j', type=int, metavar='N',
                            help='number of worker processes to use; default is # of CPU cores')

    # view subparser
    view_parser = subparsers.add_parser('view', help='view samples from connectivity dataset')
    view_parser.add_argument('dataset', type=str, help='dataset to view samples from')
    view_parser.add_argument('-s', '--samples', metavar='N', type=int, default=5,
                             help='number of samples to view')
    view_parser.add_argument('--dpi', type=int, default=150, help='dpi to use for figure')

    p = parser.parse_args()

    if p.command == 'generate':
        generate_hdf5_dataset(p.task_count, p.samples, p.jobs)
    elif p.command == 'view':
        # helps the figures to be readable on hidpi screens
        mpl.rcParams['figure.dpi'] = p.dpi
        view_hdf5_dataset(p.dataset, p.samples)
