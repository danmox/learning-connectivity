import pytorch_lightning as pl
from pathlib import Path
from hdf5_dataset_utils import kernelized_config_img, cnn_image_parameters, subs_to_pos, pos_to_subs
from math import ceil, sqrt
from cnn import BetaVAEModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from network_planner.connectivity_optimization import ConnectivityOpt as ConnOpt
from feasibility import connect_graph
import torch
from hdf5_dataset_utils import ConnectivityDataset
import h5py


def threshold(img, val):
    tmp = np.copy(img)
    tmp[tmp < val] = 0
    return tmp


def compute_blobs(img):
    blobs = blob_log(threshold(img, 40), max_sigma=7, min_sigma=5, num_sigma=10, threshold=.2)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return blobs


def connectivity_from_image(task_config, out_img, p):
    blobs = blob_log(threshold(out_img, 40), max_sigma=7, min_sigma=5, num_sigma=10, threshold=.2)
    comm_config = np.zeros((blobs.shape[0], 2))
    for i, blob in enumerate(blobs):
        comm_config[i,:] = subs_to_pos(p['meters_per_pixel'], p['img_size'][0], blob[:2])
    return ConnOpt.connectivity(p['channel_model'], task_config, comm_config)


def connectivity_from_config(task_config, p):
    comm_config = connect_graph(task_config, p['comm_range'])
    opt = ConnOpt(p['channel_model'], task_config, comm_config)
    conn = opt.maximize_connectivity()
    return conn, opt.get_comm_config()


def line_test(args):

    model_file = Path(args.model)
    if not model_file.exists():
        print(f'provided model {model_file} not found')
        return
    model = BetaVAEModel.load_from_checkpoint(args.model, beta=1.0, z_dim=16)

    params = cnn_image_parameters()

    start_config = np.asarray([[0, 20], [0, -20]])
    step = 2*np.asarray([[0, 1],[0, -1]])
    for i in [3, 5, 8, 11, 13, 16, 19]:
        task_config = start_config + i*step
        img = kernelized_config_img(task_config, params)
        out = model.inference(img)

        img_conn = connectivity_from_image(task_config, out, params)
        opt_conn, opt_comm_config = connectivity_from_config(task_config, params)
        opt_comm_subs = pos_to_subs(params['meters_per_pixel'], params['img_size'][0], opt_comm_config)

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(img)
        ax[1].imshow(out)
        if args.blobs:
            for blob in compute_blobs(out):
                y, x, r = blob
                c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
                ax[1].add_patch(c)
            ax[1].plot(opt_comm_subs[:,1], opt_comm_subs[:,0], 'rx', ms=14, markeredgewidth=2)
        fig.suptitle(f'img conn. = {img_conn:.4f}, opt conn. = {opt_conn:.4f}', fontsize=16)
        plt.tight_layout()
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
    line_parser.add_argument('--blobs', action='store_true', help='extract blobs from output image')

    worst_parser = subparsers.add_parser('worst', help='show examples where the provided model performs the worst on the given dataset')
    worst_parser.add_argument('model', type=str, help='model to test')
    worst_parser.add_argument('dataset', type=str, help='test dataset')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    if args.command == 'line':
        line_test(args)
    elif args.command == 'worst':
        worst_test(args)
