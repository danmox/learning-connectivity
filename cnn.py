#!/usr/bin/env python3

import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime
import argparse

# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_imgs(in_imgs, net_imgs, out_imgs, show=True):
    assert(in_imgs.shape[0] == net_imgs.shape[0] == out_imgs.shape[0] and net_imgs.shape == out_imgs.shape)
    num_layers = net_imgs.shape[1]
    num_rows = in_imgs.shape[0]
    num_cols = 2*num_layers+1
    for i in range(num_rows):
        in_tmp = torch.clamp(in_imgs[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        net_tmp = torch.clamp(net_imgs[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        out_tmp = torch.clamp(out_imgs[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        plt.subplot(num_rows, num_cols, i*num_cols+1)
        plt.imshow(in_tmp)
        for j in range(num_layers):
            plt.subplot(num_rows, num_cols, i*num_cols+2+j)
            plt.imshow(net_tmp[j,...])
            plt.subplot(num_rows, num_cols, i*num_cols+2+num_layers+j)
            plt.imshow(out_tmp[j,...])
    if show:
        plt.show()


class AutoEncoderCNN(nn.Module):
    def __init__(self):
        super(AutoEncoderCNN, self).__init__()

        # encoder
        self.econv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.econv2 = nn.Conv2d(4, 8, 5, padding=2)
        self.econv3 = nn.Conv2d(8, 12, 5, padding=2)
        self.econv4 = nn.Conv2d(12, 16, 5, padding=2)
        self.econv5 = nn.Conv2d(16, 20, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        # TODO need more parameters
        # TODO squeeze to vector

        # decoder
        self.dconv1 = nn.Conv2d(20, 16, 5, padding=2)
        self.dconv2 = nn.Conv2d(16, 12, 5, padding=2)
        self.dconv3 = nn.Conv2d(12, 8, 5, padding=2)
        self.dconv4 = nn.Conv2d(8, 4, 5, padding=2)
        self.dconv5 = nn.Conv2d(4, 3, 5, padding=2)

    def forward(self, x):

        # encoder
        x = self.pool(F.relu(self.econv1(x)))
        x = self.pool(F.relu(self.econv2(x)))
        x = self.pool(F.relu(self.econv3(x)))
        x = self.pool(F.relu(self.econv4(x)))
        x = self.pool(F.relu(self.econv5(x)))

        # decoder
        x = F.relu(self.dconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv3(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv4(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = F.relu(self.dconv5(F.interpolate(x, scale_factor=2, mode='nearest')))

        return x


class ImgImgCNN(nn.Module):
    def __init__(self):
        super(ImgImgCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5, padding=2)
        self.conv2 = nn.Conv2d(1, 1, 5, padding=2)
        self.conv3 = nn.Conv2d(1, 1, 5, padding=2)
        self.conv4 = nn.Conv2d(1, 1, 5, padding=2)
        self.conv5 = nn.Conv2d(1, 1, 5, padding=2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        return x


def np_to_tensor(data, shape, device=None):
    '''
    normalize [0, 255] image data to [0,1], convert it to a pytorch tensor and
    load it on to the device, if provided
    '''
    if device is None:
        return torch.from_numpy((data / 255.0).reshape(shape)).float()
    else:
        return torch.from_numpy((data / 255.0).reshape(shape)).float().to(device)


if __name__ == '__main__':

    # argparsing

    parser = argparse.ArgumentParser(description='train connectivity CNN')
    parser.add_argument('dataset', type=str, help='dataset for training / testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--model', type=str, help='model to continue training')
    parser.add_argument('--nolog', action='store_true', help='disable logging')
    parser.add_argument('--noask', action='store_true',
                        help='skip asking user to continue training beyond given number of epochs')
    p = parser.parse_args()

    # load dataset

    dataset = Path(p.dataset)
    if not dataset.exists():
        print(f'provided dataset {dataset} not found')
        exit(1)
    hdf5_file = h5py.File(dataset, mode='r')

    # initialize training device, using GPU if available

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'training with device: {device}')

    # initialize model or load an existing one

    if p.model is None:
        net = AutoEncoderCNN() # TODO randomize initialization
        print(f'initialized new network with {count_parameters(net)} parameters')
    else:
        model_file = Path(p.model)
        if not model_file.exists():
            print(f'provided model {model_file} not found')
            exit(1)
        net = AutoEncoderCNN()
        net.load_state_dict(torch.load(model_file))
        print(f'loaded model from {model_file} with {count_parameters(net)} parameters')
    net.to(device)

    # define loss and optimizer

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-6) # NOTE start low and increase

    # initialize training parameters

    train_in = hdf5_file['train']['task_img']
    train_out = hdf5_file['train']['comm_img']
    out_layers = train_out.shape[1]
    img_res = train_in.shape[1]
    batch_size = 5 # NOTE use powers of 2?
    epochs = p.epochs
    update_interval = 100  # print loss after this many training steps
    in_shape = (batch_size, 1, img_res, img_res)
    out_shape = (batch_size, out_layers, img_res, img_res)

    # train network

    if not p.nolog:
        writer = SummaryWriter()
    loss_it = 0
    total_epochs = 0
    train = True

    while train:

        # loop over the dataset multiple times

        for epoch in range(epochs):

            running_loss = 0.0
            batch = np.arange(train_in.shape[0])
            np.random.shuffle(batch)
            batch = batch.reshape((-1, batch_size))
            batch.sort()
            for i in range(batch.shape[0]):

                in_ten = np_to_tensor(train_in[batch[i],...], in_shape, device)
                out_ten = np_to_tensor(train_out[batch[i],...], out_shape, device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # predict
                net_ten = net(in_ten)

                # backprop + optimize
                loss = criterion(net_ten, out_ten)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % update_interval == (update_interval-1):
                    if not p.nolog:
                        writer.add_scalar('loss', running_loss / update_interval, loss_it)
                    loss_it += 1
                    print(f'[{total_epochs+epoch+1:3d}, {i+1:4d}] loss: {running_loss/update_interval:.5f}')
                    running_loss = 0.0

            if p.nolog:
                continue

            # show learning progress in tensorboard

            with torch.no_grad():
                in_ten = np_to_tensor(train_in[0:batch_size,...], in_shape, device)
                net_ten = net(in_ten)

                ten_list = []
                out_ten = np_to_tensor(train_out[0:batch_size,...], out_shape)
                img_shape = (1, img_res, img_res)
                for i in range(batch_size):
                    ten_list.append(in_ten[i].cpu().detach())
                    ten_list += [net_ten[i,j].cpu().detach().reshape(img_shape) for j in range(out_layers)]
                    ten_list += [out_ten[i,j].reshape(img_shape) for j in range(out_layers)]

                grid = make_grid(ten_list, nrow=2*out_layers+1, padding=20, pad_value=1)
                writer.add_image('results', grid, global_step=epoch+1)

        total_epochs += epochs
        print(f'completed {total_epochs} epochs of training')

        # ask user if they want to extend training

        if p.noask:
            break
        else:
            user_input = '-'
            while user_input not in ['y','N','n','']:
                user_input = input('continue training (y/N)?: ')
            if user_input in ['','N','n']:
                break
            else:
                while True:
                    try:
                        user_input = input('enter number of additional epochs to train: ')
                        epochs = int(user_input)
                        break
                    except:
                        print('enter a valid integer')
                        pass

    if not p.nolog:
        writer.close()
    print('Finished Training')

    # # test network on one test instance

    # test_in = hdf5_file['test']['task_img']
    # test_out = hdf5_file['test']['comm_img']
    # idx = np.random.randint(test_in.shape[0])

    # batch = np.arange(test_in.shape[0])
    # np.random.shuffle(batch)
    # batch = batch.reshape((-1, batch_size))
    # batch.sort()

    # with torch.no_grad():

    #     # convert the batch to a tensor and load it onto the GPU
    #     in_ten = np_to_tensor(test_in[batch[0],...], in_shape, device)
    #     out_ten = np_to_tensor(test_out[batch[0],...], out_shape, device)

    #     # predict
    #     net_ten = net(in_ten)

    #     # show
    #     show_imgs(in_ten, net_ten, out_ten)

    # save model

    user_input = '-'
    while user_input not in ['y','Y','n','']:
        user_input = input('save model (Y/n)?: ')
    if user_input in ['y','Y','']:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        model_file = Path(__file__).resolve().parent / 'models' / ('connectivity_' + timestamp + '.pth')
        torch.save(net.state_dict(), model_file)
        print(f'saved model to {model_file}')
    else:
        print('model not saved to file')

