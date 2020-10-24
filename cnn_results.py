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

    input_image = hdf5_file['test']['task_img'][args.sample,...]
    output_image = hdf5_file['test']['comm_img'][args.sample,...]
    model_image = model.inference(input_image)

    blurred_image = gaussian(model_image, sigma=2)
    thresh_mask = threshold_local(blurred_image, 15, method='generic', param=lambda a : max(a.max(), 0.01))
    peaks_image = blurred_image >= thresh_mask
    peaks = np.argwhere(peaks_image)

    fig, ax = plt.subplots()
    ax.imshow(model_image) #, cmap='gray')
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
    for i in [1, 3, 5, 8, 11, 13, 16, 19]:
        task_config = start_config + i*step
        img = kernelized_config_img(task_config, params)
        out = model.inference(img)

        cnn_conn, cnn_config = connectivity_from_image(task_config, out, params)
        true_conn, true_config = connectivity_from_config(task_config, params)
        true_subs = pos_to_subs(params['meters_per_pixel'], params['img_size'][0], true_config)
        cnn_subs = pos_to_subs(params['meters_per_pixel'], params['img_size'][0], cnn_config)

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].imshow(out)
        ax[1].axis('off')
        ax[1].plot(cnn_subs[:,1], cnn_subs[:,0], 'ro', label='CNN')
        ax[1].plot(true_subs[:,1], true_subs[:,0], 'bo', label='opt')
        fig.suptitle(f'img conn. = {cnn_conn:.4f}, opt conn. = {true_conn:.4f}',
                     fontsize=16, y=0.85)
        ax[1].legend()
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

    worst_parser = subparsers.add_parser('worst', help='show examples where the provided model performs the worst on the given dataset')
    worst_parser.add_argument('model', type=str, help='model to test')
    worst_parser.add_argument('dataset', type=str, help='test dataset')

    seg_parser = subparsers.add_parser('segment', help='test out segmentation method for extracting distribution from image')
    seg_parser.add_argument('dataset', type=str, help='test dataset')
    seg_parser.add_argument('model', type=str, help='model')
    seg_parser.add_argument('sample', type=int, help='sample to test')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    if args.command == 'line':
        line_test(args)
    elif args.command == 'worst':
        worst_test(args)
    elif args.command == 'segment':
        segment_test(args)
