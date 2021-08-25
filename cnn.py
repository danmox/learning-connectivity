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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from hdf5_dataset_utils import ConnectivityDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_imgs(x, z, y, show=True):
    """
    plot results of model

    inputs:
      x    - the input to the network
      z    - the output of the network
      y    - the desired output
      show - whether or not to call the blocking show function of matplotlib
    """
    assert(x.shape[0] == z.shape[0] == y.shape[0] and z.shape == y.shape)
    layers = z.shape[1]
    rows = x.shape[0]
    cols = 2*layers+1
    for i in range(rows):
        x_tmp = torch.clamp(x[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        z_tmp = torch.clamp(z[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        y_tmp = torch.clamp(y[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        plt.subplot(rows, cols, i*cols+1)
        plt.imshow(x_tmp)
        plt.subplot(rows, cols, i*cols+2)
        plt.imshow(z_tmp)
        plt.subplot(rows, cols, i*cols+3)
        plt.imshow(y_tmp)
    if show:
        plt.show()


class View(nn.Module):
    """helper class to provide view functionality in sequential containers"""
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AEBase(pl.LightningModule):
    """base class for connectivity autoencoder models"""

    def __init__(self, log_step):
        super().__init__()

        # cache some data to show learning progress
        self.train_progress_batch = None
        self.val_progress_batch = None

        # want logging frequency less than every training iteration and more
        # than every epoch
        self.log_step = log_step
        self.loss_hist = 0.0
        self.log_it = 0

        # keep track of the best validation models
        self.val_loss_best = float('Inf')
        self.best_model_path = None

        self.model_name = ''

    # log network output for a single batch
    def log_network_image(self, batch, name):
        x, y = batch
        y_hat = self.output(x)

        img_list = []
        for i in range(x.shape[0]):
            img_list.append(x[i,...].cpu().detach())
            img_list.append(y_hat[i,...].cpu().detach())
            img_list.append(y[i,...].cpu().detach())

        grid = torch.clamp(make_grid(img_list, nrow=3, padding=20, pad_value=1), 0.0, 1.0)
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def _ckpt_dir(self):
        log_dir = Path(self.trainer.weights_save_path) / self.logger.save_dir
        return log_dir / f'version_{self.logger.version}' / 'checkpoints'

    def training_step(self, batch, batch_idx):
        # set aside some data to show learning progress
        if self.train_progress_batch is None:
            self.train_progress_batch = batch

        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.loss_hist += loss.item()
        if batch_idx != 0 and batch_idx % self.log_step == 0:
            self.logger.experiment.add_scalar('loss', self.loss_hist / self.log_step, self.log_it)
            self.loss_hist = 0.0
            self.log_it += 1

        return loss

    # provide visual feedback of the learning progress after every epoch
    def training_epoch_end(self, outs):
        torch.set_grad_enabled(False)
        self.eval()

        self.log_network_image(self.train_progress_batch, 'train_results')

        torch.set_grad_enabled(True)
        self.train()

    def validation_step(self, batch, batch_idx):
        # set aside some data to show network performance on validation data
        if self.val_progress_batch is None:
            self.val_progress_batch = batch

        x, y = batch
        y_hat = self.output(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_epoch_end(self, outs):
        val_loss = np.mean(np.asarray([o.item() for o in outs]))

        # save checkpoint of best performing model
        if val_loss < self.val_loss_best:
            self.val_loss_best = val_loss

            if self.best_model_path is not None:
                self.best_model_path.unlink(missing_ok=True)

            filename = self._ckpt_dir() / (self.model_name + f'valloss_{val_loss:.3e}_epoch_{self.current_epoch}.ckpt')
            self.trainer.save_checkpoint(filename, weights_only=True)
            self.best_model_path = filename

        self.logger.experiment.add_scalar('val_loss', val_loss, self.current_epoch)
        self.log_network_image(self.val_progress_batch, 'val_results')

    def output(self, x):
        return self(x)

    def evaluate(self, x):
        with torch.no_grad():
            x = x[None, None] / 255.0
            y_hat = self(x.float())
            y_hat = torch.clamp(255*y_hat, 0, 255).to(torch.uint8).squeeze()
        return y_hat

    def inference(self, x):
        """perform inference on model from numpy kernelized config image

        inputs:
          x - a numpy image generated from kernelized_config_img

        outputs:
          y_hat - a numpy image in the same format as x
        """
        return self.evaluate(torch.from_numpy(x)).cpu().detach().numpy()


class UAEModel(AEBase):
    """your stock undercomplete auto encoder"""

    @classmethod
    def parse_model_name(cls, name):
        parts = name.split('_')
        assert parts[0] == 'uae'
        assert parts[1] == 'nf'
        args = {}
        args['nf'] = int(parts[2])
        return args

    def __init__(self, log_step=1, nf=32):
        super().__init__(log_step)

        self.model_name = f'uae_nf_{nf}_'

        self.network = nn.Sequential(            #  1, 256, 256 (input)
            nn.Conv2d(1, nf, 8, 2, 3),           # nf, 128, 128
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 8, 2, 3),          # nf,  64,  64
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  32,  32
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.ReLU(True),
            View((-1, nf*4*4)),                  # nf*16
            nn.Linear(nf*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, nf*4*4),              # nf*16
            nn.ReLU(True),
            View((-1, nf, 4, 4)),                # nf,   4,   4
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 8, 2, 3), # nf, 128, 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 8, 2, 3)   # nf, 256, 256
        )                                        #  1, 256, 256 (output)

        # initialize weights
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class BetaVAEModel(AEBase):
    """Beta Variational Auto Encoder class for learning connectivity from images

    based on BetaVAE_B from https://github.com/1Konny/Beta-VAE/blob/master/model.py
    which is based on: https://arxiv.org/abs/1804.03599

    """

    @classmethod
    def parse_model_name(cls, name):
        parts = name.split('_')
        assert parts[0] == 'betavae'
        assert parts[1] == 'beta'
        assert parts[3] == 'kld'
        assert parts[5] == 'z'
        assert parts[7] == 'nf'
        args = {}
        args['beta'] = float(parts[2])
        args['kld_weight'] = float(parts[4])
        args['z_dim'] = int(parts[6])
        args['nf'] = int(parts[8])
        return args

    def __init__(self, beta, z_dim, kld_weight=1.0, log_step=1, nf=32):
        super().__init__(log_step)

        self.model_name = f'betavae_beta_{beta}_kld_{kld_weight}_z_{z_dim}_nf_{nf}_'

        self.beta = beta
        self.kld_weight = kld_weight
        self.z_dim = z_dim # dimension of latent distribution

        self.encoder = nn.Sequential(            #  1, 256, 256 (input)
            nn.Conv2d(1, nf, 4, 2, 1),           # nf, 128, 128
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  64,  64
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  32,  32
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.ReLU(True),
            View((-1, nf*4*4)),                  # nf*16
            nn.Linear(nf*4*4, 256),              # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, 2*z_dim)              # 2 x z_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # 256
            nn.ReLU(True),
            nn.Linear(256, nf*4*4),              # nf*16
            nn.ReLU(True),
            View((-1, nf, 4, 4)),                # nf,   4,   4
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf, 128, 128
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 4, 2, 1)   # nf, 256, 256
        )                                        #  1, 256, 256 (output)

        # initialize weights
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):

        # encoder spits out latent distribution as a single 32x1 vector with the
        # first 16 elements corresponding to the mean and the last 16 elements
        # corresponding to the log of the variance
        latent_distribution = self.encoder(x)
        mu = latent_distribution[:, :self.z_dim]
        logvar = latent_distribution[:, self.z_dim:]

        if self.training:
            # generate the input to the decoder using the reparameterization trick
            std = logvar.div(2).exp()
            eps = torch.randn_like(std)
            z = mu + std*eps
        else:
            z = mu

        out = self.decoder(z)

        return out, mu, logvar

    def output(self, x):
        return self(x)[0]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # set aside some data to show learning progress on training data
        if self.train_progress_batch is None:
            self.train_progress_batch = batch

        x, y = batch
        y_hat, mu, logvar = self(x)

        recon_loss = F.mse_loss(y_hat, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + self.beta * self.kld_weight * kld_loss

        self.loss_hist += loss.item()
        if batch_idx != 0 and batch_idx % self.log_step == 0:
            self.logger.experiment.add_scalar('loss', self.loss_hist / self.log_step, self.log_it)
            self.loss_hist = 0.0
            self.log_it += 1

        return loss

    def evaluate(self, x):
        with torch.no_grad():
            x = x[None, None] / 255.0
            y_hat, _, _ = self(x.float())
            y_hat = torch.clamp(255*y_hat, 0, 255).to(torch.uint8).squeeze()
        return y_hat


class ConvAEModel(AEBase):
    """convolutions all the way"""

    @classmethod
    def parse_model_name(cls, name):
        parts = name.split('_')
        assert parts[0] == 'convae'
        assert parts[1] == 'nf'
        args = {}
        args['nf'] = int(parts[2])
        return args

    def __init__(self, log_step=1, nf=64):
        super().__init__(log_step)

        self.model_name = f'convae_nf_{nf}_'

        self.network = nn.Sequential(            #  1, 128, 128 (input)
            nn.Conv2d(1, nf, 8, 2, 3),           # nf,  64,  64
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 8, 2, 3),          # nf,  32,  32
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,  16,  16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   8,   8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1),          # nf,   4,   4
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,   8,   8
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 4, 2, 1), # nf,  32,  32
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nf, 8, 2, 3), # nf,  64,  64
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, 1, 8, 2, 3),  # nf, 128, 128
        )                                        #  1, 128, 128 (output)

        # initialize weights
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


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

    if args.type == 'betavae':

        beta = 1.0
        z_dim = 16
        kld_weight = 1.0 / len(train_dataloader)

        if args.model is not None:
            model_path = Path(args.model)
            if not model_path.exists():
                print(f'provided model {model_path} not found')
                return
            model = BetaVAEModel.load_from_checkpoint(args.model, beta=beta, z_dim=z_dim,
                                                      kld_weight=kld_weight, log_step=log_step)
        else:
            model = BetaVAEModel(beta=beta, z_dim=z_dim, kld_weight=kld_weight, log_step=log_step)

    elif args.type == 'convae':

        if args.model is not None:
            model_path = Path(args.model)
            if not model_path.exists():
                print(f'provided model {model_path} not found')
                return
            model = ConvAEModel.load_from_checkpoint(args.model, log_step=log_step)
        else:
            model = ConvAEModel(log_step=log_step)

    elif args.type == 'uae':

        if args.model is not None:
            model_path = Path(args.model)
            if not model_path.exists():
                print(f'provided model {model_path} not found')
                return
            model = UAEModel.load_from_checkpoint(args.model, log_step=log_step)
        else:
            model = UAEModel(log_step=log_step)

    # train network

    logger = pl_loggers.TensorBoardLogger('runs/', name='')
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, weights_summary='top', gpus=gpus)
    trainer.fit(model, train_dataloader, val_dataloader)


def get_file_name(path):
    """return name of filename, following symlinks if necessary"""
    filename = Path(path)
    if filename.is_symlink():
        return Path(os.readlink(filename)).stem
    else:
        return filename.stem


def load_model_for_eval(model_file):
    model_file = Path(model_file)
    if not model_file.exists():
        print(f'provided model {model_file} not found')
        return None

    # be sure to follow symlinks before parsing filename
    model_file_name = get_file_name(model_file)

    model_type = model_file_name.split('_')[0]
    if model_type == 'betavae':
        args = BetaVAEModel.parse_model_name(model_file_name)
        model = BetaVAEModel.load_from_checkpoint(str(model_file), **args)
    elif model_type == 'convae':
        args = ConvAEModel.parse_model_name(model_file_name)
        model = ConvAEModel.load_from_checkpoint(str(model_file), **args)
    elif model_type == 'uae':
        args = UAEModel.parse_model_name(model_file_name)
        model = UAEModel.load_from_checkpoint(str(model_file), **args)
    else:
        print(f'unrecognized model type {model_type} from model {model_file}')
        return None
    model.eval()
    return model


def eval_main(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

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

    if not args.save:
        print(f'showing sample {idx} from {dataset_file.name}')
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

    if args.save:
        imgs = (input_image, output_image, model_image)
        names = ('input', 'output', 'model')
        for img, name in zip(imgs, names):
            fig = plt.figure()
            fig.set_size_inches((4,4))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(img, aspect='equal')
            filename = '_'.join((str(idx), name, dataset_file.stem)) + '.png'
            plt.savefig(filename, dpi=150)
            print(f'saved image {filename}')

    hdf5_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='utilities for train and testing a connectivity CNN')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # train subparser
    train_parser = subparsers.add_parser('train', help='train connectivity CNN model on provided dataset')
    train_parser.add_argument('type', type=str, help='model type to train', choices=['uae', 'betavae', 'convae'])
    train_parser.add_argument('dataset', type=str, help='dataset for training', nargs='+')
    train_parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    train_parser.add_argument('--batch-size', type=int, default=4, help='batch size for training')
    train_parser.add_argument('--model', type=str, help='checkpoint to load')

    # inference subparser
    eval_parser = subparsers.add_parser('eval', help='run inference on samples(s)')
    eval_parser.add_argument('model', type=str, help='model to use for inference')
    eval_parser.add_argument('dataset', type=str, help='dataset to draw samples from')
    eval_parser.add_argument('--sample', type=int, help='sample to perform inference on')
    eval_parser.add_argument('--save', action='store_true', help='save intput, output and target images')

    args = parser.parse_args()

    if args.command == 'train':
        train_main(args)
    if args.command == 'eval':
        eval_main(args)
