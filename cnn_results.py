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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN network tests')
    parser.add_argument('model', type=str, help='model to test')
    parser.add_argument('--blobs', action='store_true', help='extract blobs from output image')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    line_test(args)
