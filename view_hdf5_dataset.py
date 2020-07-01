#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import h5py
import argparse
import json


# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


if __name__ == '__main__':

    # parse input

    parser = argparse.ArgumentParser(description='generate dataset for learning connectivity')
    parser.add_argument('--dataset', type=str, help=f'Dataset to convert', required=True)
    parser.add_argument('--samples', type=int, default=10, help='number of samples of dataset to display')
    p = parser.parse_args()

    # init

    dataset = Path(p.dataset)
    if not dataset.exists():
        print(f'the dataset {dataset} was not found')
        exit(-1)
    params_file = Path(p.dataset).with_name(dataset.stem + '.json')
    if not params_file.exists():
        print(f'the parameter file {str(params_file)} was not found')
        exit(-1)
    with open(params_file, 'r') as f:
        params = json.load(f)

    hdf5_file = h5py.File(dataset, mode='r')

    sample_counts = [hdf5_file[m]['init_img'].shape[0] for m in ('train','test')]
    sample_idcs = [np.random.randint(0, m, size=(min(m, p.samples),)) for m in sample_counts]
    for idcs, mode in zip(sample_idcs, ('train','test')):
        idcs.sort()
        print(f"plotting {len(idcs)} {mode}ing samples: {', '.join(map(str, idcs))}")
        for i, idx in enumerate(idcs):
            task_config = hdf5_file[mode]['task_config'][idx,...]
            combined_config = np.vstack((task_config, hdf5_file[mode]['comm_config'][idx,...]))
            ax = plt.subplot(1,4,1)
            ax.plot(task_config[:,0], task_config[:,1], 'g.', ms=4)
            ax.axis('scaled')
            ax.axis(params['bbx'])
            plt.subplot(1,4,2)
            plt.imshow(hdf5_file[mode]['init_img'][idx,...])
            ax = plt.subplot(1,4,3)
            ax.plot(combined_config[:,0], combined_config[:,1], 'r.', ms=4)
            ax.axis('scaled')
            ax.axis(params['bbx'])
            plt.subplot(1,4,4)
            plt.imshow(hdf5_file[mode]['final_img'][idx,...])
            plt.suptitle(f'sample {i+1}/{len(idcs)} with index {idx}', fontsize=14)
            plt.show()

    hdf5_file.close()
