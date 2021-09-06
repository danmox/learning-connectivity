#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from math import ceil

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from torch.utils.data import DataLoader

from hdf5_dataset_utils import ConnectivityDataset, cnn_image_parameters
from models import *
from mid.connectivity_planner.src.connectivity_planner.feasibility import connect_graph


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_file_name(path):
    """return name of filename, following symlinks if necessary"""
    filename = Path(path)
    if filename.is_symlink():
        return Path(os.readlink(filename)).stem
    else:
        return filename.stem


def load_model_from_checkpoint(model_file):
    """load model from saved checkpoint"""
    model_file = Path(model_file)
    if not model_file.exists():
        print(f'provided model {model_file} not found')
        return None

    # be sure to follow symlinks before parsing filename
    model_file_name = get_file_name(model_file)

    model_type = model_file_name.split('__')[0]
    try:
        model = globals()[model_type].load_from_checkpoint(str(model_file))
    except:
        print(f'unrecognized model type {model_type} from model {model_file}')
        return None

    return model


def load_model_for_eval(model_file):
    """load saved model for evaluation"""

    model = load_model_from_checkpoint(model_file)

    if model is None:
        return None

    model.eval()
    return model


def train_main(args):

    cpus = os.cpu_count()
    gpus = 1 if torch.cuda.is_available() else 0

    # load dataset

    train_dataset = ConnectivityDataset(args.dataset, train=True)
    val_dataset = ConnectivityDataset(args.dataset, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=cpus)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=cpus)
    dataset_names = '\n'.join(args.dataset)
    print('training on the following dataset(s):')
    print(f'{dataset_names}')

    # initialize model or load one, if provided

    log_step = ceil(len(train_dataloader)/100) # log averaged loss ~100 times per epoch
    kld_weight = 1.0 / len(train_dataloader)

    if args.model[-5:] == '.ckpt':
        model = load_model_from_checkpoint(args.model)
        if model is None:
            return
    else:
        try:
            model = globals()[args.model](log_step=log_step, kld_weight=kld_weight)
        except:
            print(f'unrecognized model type {args.model}')
            return

    # train network

    logger = pl_loggers.TensorBoardLogger('runs/', name=model.model_name)
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, weights_summary='top', gpus=gpus)
    trainer.fit(model, train_dataloader, val_dataloader)


def eval_main(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')

    mode = 'train' if args.train else 'test'
    dataset_len = hdf5_file[mode]['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of {mode} dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    input_image = hdf5_file[mode]['task_img'][idx,...]
    output_image = hdf5_file[mode]['comm_img'][idx,...]
    model_image = model.inference(input_image)

    if args.arrays:
        task_config = hdf5_file[mode]['task_config'][idx,...]
        print(f'task_config:\n{task_config}')
        mst_config = connect_graph(task_config, cnn_image_parameters()['comm_range'])
        print(f'msg_config:\n{mst_config}')
        comm_config = hdf5_file[mode]['comm_config'][idx,...]
        print(f'comm_config:\n{comm_config[~np.isnan(comm_config[:,0])]}')

    if not args.save:
        print(f'showing sample {idx} from {dataset_file.name}')
        ax = plt.subplot(1,3,1)
        ax.imshow(input_image.T)
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('input')
        ax = plt.subplot(1,3,2)
        ax.imshow(output_image.T)
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('target')
        ax = plt.subplot(1,3,3)
        ax.imshow(model_image.T)
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('output')
        plt.tight_layout()
        plt.show()

    if args.save:
        imgs = (input_image, output_image, model_image)
        names = ('input', 'output', 'model')
        for img, name in zip(imgs, names):
            fig = plt.figure()
            fig.set_size_inches((4,4))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(np.flipud(img.T), aspect='equal', cmap='Greys')
            filename = '_'.join((str(idx), name, dataset_file.stem)) + '.png'
            plt.savefig(filename, dpi=150)
            print(f'saved image {filename}')

    hdf5_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='utilities for train and testing a connectivity CNN')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # train subparser
    train_parser = subparsers.add_parser('train', help='train connectivity CNN model on provided dataset')
    train_parser.add_argument('model', type=str, help='model type to train or checkpoint to pick up from')
    train_parser.add_argument('dataset', type=str, help='dataset for training', nargs='+')
    train_parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    train_parser.add_argument('--batch-size', type=int, default=4, help='batch size for training')

    # inference subparser
    eval_parser = subparsers.add_parser('eval', help='run inference on samples(s)')
    eval_parser.add_argument('model', type=str, help='model to use for inference')
    eval_parser.add_argument('dataset', type=str, help='dataset to draw samples from')
    eval_parser.add_argument('--sample', type=int, help='sample to perform inference on')
    eval_parser.add_argument('--save', action='store_true', help='save intput, output and target images')
    eval_parser.add_argument('--train', action='store_true', help='select sample from training partition')
    eval_parser.add_argument('--arrays', action='store_true', help='print task/comm. agent arrays')

    args = parser.parse_args()

    if args.command == 'train':
        train_main(args)
    if args.command == 'eval':
        eval_main(args)
