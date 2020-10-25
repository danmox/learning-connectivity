import pytorch_lightning as pl
from pathlib import Path
from hdf5_dataset_utils import kernelized_config_img, cnn_image_parameters, subs_to_pos, pos_to_subs
from math import ceil, sqrt
from cnn import BetaVAEModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import numpy as np
from skimage.filters.thresholding import threshold_local
from skimage.filters import gaussian
from network_planner.connectivity_optimization import ConnectivityOpt as ConnOpt
from feasibility import connect_graph
import torch
from hdf5_dataset_utils import ConnectivityDataset
import h5py


def compute_peaks(image):
    blurred_image = gaussian(image, sigma=2)
    thresh_mask = threshold_local(blurred_image, 11, method='generic', param=lambda a : max(a.max(), 0.01))
    peaks = np.argwhere(blurred_image >= thresh_mask)
    return peaks


def connectivity_from_image(task_config, out_img, p):
    peaks = compute_peaks(out_img)
    comm_config = np.zeros_like(peaks)
    for i, peak in enumerate(peaks):
        comm_config[i,:] = subs_to_pos(p['meters_per_pixel'], p['img_size'][0], peak)
    connectivity = ConnOpt.connectivity(p['channel_model'], task_config, comm_config)
    return connectivity, comm_config


def connectivity_from_config(task_config, p):
    comm_config = connect_graph(task_config, p['comm_range'])
    opt = ConnOpt(p['channel_model'], task_config, comm_config)
    conn = opt.maximize_connectivity()
    return conn, opt.get_comm_config()


def segment_test(args):

    model_file = Path(args.model)
    if not model_file.exists():
        print(f'provided model {model_file} not found')
        return
    model = BetaVAEModel.load_from_checkpoint(args.model, beta=1.0, z_dim=16)
    model.eval()

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file['test']['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    input_image = hdf5_file['test']['task_img'][idx,...]
    output_image = hdf5_file['test']['comm_img'][idx,...]
    model_image = model.inference(input_image)

    blurred_image = gaussian(model_image, sigma=2)
    thresh_mask = threshold_local(blurred_image, 15, method='generic', param=lambda a : max(a.max(), 0.01))
    peaks_image = blurred_image >= thresh_mask
    peaks = np.argwhere(peaks_image)

    fig, ax = plt.subplots()
    ax.imshow(np.maximum(model_image, input_image))
    ax.axis('off')
    ax.plot(peaks[:,1], peaks[:,0], 'ro')
    plt.show()


def line_test(args):

    model_file = Path(args.model)
    if not model_file.exists():
        print(f'provided model {model_file} not found')
        return
    model = BetaVAEModel.load_from_checkpoint(args.model, beta=1.0, z_dim=16)

    params = cnn_image_parameters()

    start_config = np.asarray([[0, 20], [0, -20]])
    step = 2*np.asarray([[0, 1],[0, -1]])
    for i in range(20):
        task_config = start_config + i*step
        img = kernelized_config_img(task_config, params)
        out = model.inference(img)

        cnn_conn, cnn_config = connectivity_from_image(task_config, out, params)
        opt_conn, opt_config = connectivity_from_config(task_config, params)
        opt_subs = pos_to_subs(params['meters_per_pixel'], params['img_size'][0], opt_config)
        cnn_subs = pos_to_subs(params['meters_per_pixel'], params['img_size'][0], cnn_config)

        p = cnn_image_parameters()
        img_extents = p['img_side_len'] / 2.0 * np.asarray([-1,1,1,-1])

        fig, ax = plt.subplots()

        ax.imshow(np.maximum(out, img), extent=img_extents)
        ax.plot(task_config[:,1], task_config[:,0], 'ro', label='task')
        ax.plot(opt_config[:,1], opt_config[:,0], 'rx', label='comm. opt.', ms=9, mew=3)
        ax.plot(cnn_config[:,1], cnn_config[:,0], 'bx', label='comm. CNN', ms=9, mew=3)

        ax.invert_yaxis()
        ax.set_yticks(np.arange(-80, 80, 20))
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.legend(loc='best', fontsize=14)
        ax.set_title(f'opt. = {opt_conn:.3f}, cnn = {cnn_conn:.3f}', fontsize=18)

        plt.tight_layout()

        if args.save:
            filename = f'line_{i:02d}_{model_file.stem}.png'
            plt.savefig(filename, dpi=150)
            print(f'saved image {filename}')
        else:
            plt.show()


def worst_test(args):

    model_file = Path(args.model)
    if not model_file.exists():
        print(f'provided model {model_file} not found')
        return
    model = BetaVAEModel.load_from_checkpoint(args.model, beta=1.0, z_dim=16)
    model.eval()

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    dataset = ConnectivityDataset(dataset_file, train=False)

    worst_loss = 0.0
    with torch.no_grad():
        print(f'looping through {len(dataset)} test samples in {dataset_file}')
        for i in range(len(dataset)):
            print(f'\rprocessing sample {i+1} of {len(dataset)}\r', end="")
            batch = [torch.unsqueeze(ten, 0) for ten in dataset[i]]
            loss = model.validation_step(batch, None)
            if loss > worst_loss:
                worst_idx = i
                worst_loss = loss

    hdf5_file = h5py.File(dataset_file, mode='r')
    input_image = hdf5_file['test']['task_img'][worst_idx,...]
    output_image = hdf5_file['test']['comm_img'][worst_idx,...]
    model_image = model.inference(input_image)

    print(f'worst sample is {worst_idx}{20*" "}')
    ax = plt.subplot(1,3,1)
    ax.imshow(input_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('input')
    ax = plt.subplot(1,3,2)
    ax.imshow(output_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('target')
    ax = plt.subplot(1,3,3)
    ax.imshow(model_image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('output')
    plt.tight_layout()
    plt.show()

    hdf5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN network tests')
    subparsers = parser.add_subparsers(dest='command', required=True)

    line_parser = subparsers.add_parser('line', help='run line test on provided model')
    line_parser.add_argument('model', type=str, help='model to test')
    line_parser.add_argument('--save', action='store_true')

    worst_parser = subparsers.add_parser('worst', help='show examples where the provided model performs the worst on the given dataset')
    worst_parser.add_argument('model', type=str, help='model to test')
    worst_parser.add_argument('dataset', type=str, help='test dataset')

    seg_parser = subparsers.add_parser('segment', help='test out segmentation method for extracting distribution from image')
    seg_parser.add_argument('model', type=str, help='model')
    seg_parser.add_argument('dataset', type=str, help='test dataset')
    seg_parser.add_argument('--sample', type=int, help='sample to test')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    if args.command == 'line':
        line_test(args)
    elif args.command == 'worst':
        worst_test(args)
    elif args.command == 'segment':
        segment_test(args)
