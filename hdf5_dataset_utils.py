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

    def __init__(self, paths, train=True):
        # paths can be some collection of hdf5 files or paths containing hdf5
        # files
        if not isinstance(paths, list):
            paths = [paths]
        paths = [Path(f) for f in paths]

        # build unique list of hdf5 files
        self.hdf5_files = set()
        for item in paths:
            if item.is_dir():
                self.hdf5_files.update(item.glob('*.hdf5'))
            else:
                self.hdf5_files.add(item)
        assert len(self.hdf5_files) > 0

        self.mode = 'train' if train else 'test'
        self.datasets = None

        # compute dataset splits for multi-array indexing
        dataset_lens = []
        for hdf5_file in self.hdf5_files:
            with h5py.File(hdf5_file, 'r') as f:
                dataset_lens.append(f[self.mode]['connectivity'].shape[0])
        self.splits = np.hstack(([0], np.cumsum(dataset_lens)))

    def __getitem__(self, idx):
        # in order to use the multiprocessing capabilities provided by pytorch,
        # the hdf5 file must be loaded after __init__ since opened hdf5 files
        # are not pickleable:
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.datasets is None:
            self.datasets = [h5py.File(f, 'r')[self.mode] for f in self.hdf5_files]

        # determine the dataset and sample to draw
        dataset_idx = next(i for i in range(len(self.splits)) if self.splits[i] > idx) - 1
        sample_idx = idx - self.splits[dataset_idx]

        # rescale: [0,255] -> [0,1] and reshape: (X,X) -> (1, X, X)
        x = np.expand_dims(self.datasets[dataset_idx]['task_img'][sample_idx,...] / 255.0, axis=0)
        y = np.expand_dims(self.datasets[dataset_idx]['comm_img'][sample_idx,...] / 255.0, axis=0)

        # convert to float tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return (x, y)

    def __len__(self):
        return self.splits[-1]


def pos_to_subs(res, pts):
    # assumes origin is at (0,0) and x,y res is equal
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
    """generate kernelized image from agents configuration

    inputs:
      config - Nx2 numpy array of agent positions
      params - image parameters from cnn_image_parameters()

    ourputs:
      img - image with node positions marked with a gaussian kernel
    """
    img = np.zeros(params['img_size'])
    for agent in config:
        dist = np.linalg.norm(params['xy'] - agent, axis=2)
        mask = dist < 3.0*params['kernel_std']
        img[mask] = np.maximum(img[mask], norm.pdf(dist[mask], scale=params['kernel_std']))
    img *= 255.0 / norm.pdf(0, scale=params['kernel_std']) # normalize image to [0.0, 255.0]
    return np.clip(img, 0, 255)


def write_hdf5_image_data(params, filename, queue):
    """
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
    """

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
    """
    helper function for generating hdf5 training samples in a multiprocessing
    scenario

    inputs:
      params       - dict of parameters used to generate training data
      sample_queue - the queue of seed data to generate samples from
      writer_queue - the queue of data to be written to the hdf5 database
    """

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


def cnn_image_parameters():

    p = {}
    p['max_task_agents'] = 15 # the number of task agents the image is sized for
    p['comm_range'] = 30      # maximum range of communication hardware
    img_res = 128        # cnn images are img_res x img_res
    p['img_size'] = (img_res, img_res)

    # ratio of max area covered by N agents vs area of image for bbx; this controls
    # the density of the sampled task agent configurations, with highter numbers
    # leading to sparser configurations
    p['area_scale_factor'] = 0.5
    p['kernel_std'] = img_res * 0.05

    # the pysical space the image represents is determined based on the desired
    # density of the agents with some padding added to ensure that the kernel
    # used to mark each agent is not cut off by the side of the image
    img_bbx = adaptive_bbx(p['max_task_agents'], p['comm_range'], p['area_scale_factor'])
    p['space_side_length'] = 2.0*ceil(img_bbx.max() + p['kernel_std'])
    p['meters_per_pixel'] = p['space_side_length'] / img_res

    # useful for painting the kernel on the image
    px_range = np.arange(img_res)
    ij = np.stack(np.meshgrid(px_range, px_range, indexing='ij'), axis=2)
    p['xy'] = p['meters_per_pixel'] * (ij + 0.5) - p['space_side_length']/2.0

    return p


def generate_hdf5_dataset(task_agents, samples, jobs):

    params = cnn_image_parameters() # fixed image generation parameters
    params['channel_model'] = PiecewisePathLossModel(print_values=False)
    params['task_agents'] = task_agents # so that all necessary args are in params

    train_samples = int(0.85 * samples)
    params['sample_count'] = {'train': train_samples, 'test': samples - train_samples}

    # focus samples within a subset of the entire image so that interesting
    # configurations are generated (ones that differ significantly from the MST
    # solution) TODO vary this within the dataset
    sample_bbx = adaptive_bbx(task_agents, params['comm_range'], params['area_scale_factor'])

    # since sample_bbx is a subset of the entire image, generated
    # configurations for smaller numbers of task agents will only take up a
    # small portion of the image centered on the center; while convolutions are
    # shift invariant we want to ensure that all of the image is used during
    # training. thus generated configurations are shifted around the image but
    # constrainted to remain within this img_bbx
    img_bbx = (params['space_side_length']/2.0 - params['kernel_std']) * np.asarray([-1,1,-1,1])

    # an overestimate of the maximum number of task agents that might be
    # deployed used for hdf5 data sizing, which must be fixed for all samples
    # stored in the dataset
    params['max_comm_agents'] = 3 * ceil((sample_bbx[1] - sample_bbx[0]) / params['comm_range'])

    if task_agents > params['max_task_agents']:
        print(f"too many task agents ({task_agents}): parameters tuned for {params['max_task_agents']}")
        return

    # generate descriptive filename
    # NOTE there is a risk of overwriting a database if this script is run more
    # than once in a second
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = Path(__file__).resolve().parent / 'data' / \
        f"connectivity_{samples}s_{task_agents}t_{timestamp}.hdf5"

    # print dataset info to console
    res = params['img_size'][0]
    print(f"using {res}x{res} images with {params['meters_per_pixel']} meters/pixel "
          f"and {task_agents} task agents")

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
    for mode in ('train','test'):
        it = 0
        while it < params['sample_count'][mode]:
            x_task, x_comm = min_feasible_sample(task_agents, params['comm_range'], sample_bbx)
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

    # parameters related to the dataset
    params = cnn_image_parameters()

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
