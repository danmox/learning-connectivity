#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
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

# helps the figures to be readable on hidpi screens
mpl.rcParams['figure.dpi'] = 200


def show_imgs(in_imgs, net_imgs, out_imgs, show=True):
    assert(in_imgs.shape[0] == out_imgs.shape[0] and net_imgs.shape[0] == in_imgs.shape[0])
    num_imgs = in_imgs.shape[0]
    for i in range(num_imgs):
        in_tmp = torch.clamp(in_imgs[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        net_tmp = torch.clamp(net_imgs[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        out_tmp = torch.clamp(out_imgs[i] * 255.0, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()
        plt.subplot(3, num_imgs, i+1)
        plt.imshow(in_tmp)
        plt.subplot(3, num_imgs, i+1+num_imgs)
        plt.imshow(net_tmp)
        plt.subplot(3, num_imgs, i+1+num_imgs*2)
        plt.imshow(out_tmp)
    if show:
        plt.show()


# load data


dataset = Path(__file__).resolve().parent / 'data' / 'connectivity_from_imgs_10000_kernel_io.hdf5'
hdf5_file = h5py.File(dataset, mode='r')


# define networks


class AutoEncoderCNN(nn.Module):
    def __init__(self):
        super(AutoEncoderCNN, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 4, 5, padding=2),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(4, 8, 5, padding=2),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(8, 12, 5, padding=2),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(12, 8, 5, padding=2)
        #     nn.Conv2d(8, 4, 5, padding=2)
        #     nn.Conv2d(4, 1, 5, padding=2)
        # )

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
        self.dconv5 = nn.Conv2d(4, 1, 5, padding=2)

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


    # TODO standard NN on vectors of node positions
    # TODO simpler images: 2 task 1 comm


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


# TODO try alex net

# use GPU


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# TODO random initilization
net = AutoEncoderCNN()
# net = ImgImgCNN()
print(net)
net.to(device)


# define loss and optimizer


# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)
# NOTE start low and increase
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.09)
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.009)


# train network


train_in = hdf5_file['train']['task_img']
train_out = hdf5_file['train']['comm_img']
img_res = train_in.shape[1]
batch_size = 5 # NOTE use powers of 2
epochs = 100
update_interval = 100  # print loss after this many training steps

def scale_img(img):
    return img / 255.0

loss_vec = []
writer = SummaryWriter()
loss_it = 0
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    batch = np.arange(train_in.shape[0])
    np.random.shuffle(batch)
    batch = batch.reshape((-1, batch_size))
    batch.sort()
    for i in range(batch.shape[0]):

        # normalize data from [0,1] to [-1,1] and reshape to (batch_size, channels, res_x, rez_y)
        in_tmp = scale_img(train_in[batch[i],...]).reshape((batch_size, 1, img_res, img_res))
        out_tmp = scale_img(train_out[batch[i],...]).reshape((batch_size, 1, img_res, img_res))

        # convert the batch to a tensor and load it onto the GPU
        in_imgs = torch.from_numpy(in_tmp).float().to(device)
        ref_imgs = torch.from_numpy(out_tmp).float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # predict
        net_imgs = net(in_imgs)
        # if i == batch.shape[0]-1 and epoch % 2 == 0:
        #     show_imgs(in_imgs, net_imgs, ref_imgs)

        # backprop + optimize
        loss = criterion(net_imgs, ref_imgs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % update_interval == (update_interval-1):
            writer.add_scalar('loss', running_loss / update_interval, loss_it)
            loss_it += 1
            loss_vec.append(running_loss/update_interval)
            print(f'[{epoch+1:3d}, {i+1:4d}] loss: {running_loss/update_interval:.5f}')
            running_loss = 0.0

    # print(f'done: epoch {epoch+1:2d} of {epochs}')

writer.close()

print('Finished Training')


# save model

timestamp = datetime.now().strftime('%Y%m%d-%H%M')
model_file = Path(__file__).resolve().parent / 'models' / ('connectivity_' + timestamp + '.pth')
torch.save(net.state_dict(), model_file)
print(f'saved model to {model_file}')


# test network on one test instance


net = AutoEncoderCNN()
net.load_state_dict(torch.load(model_file))
net.to(device)

test_in = hdf5_file['test']['task_img']
test_out = hdf5_file['test']['comm_img']
idx = np.random.randint(test_in.shape[0])

batch = np.arange(test_in.shape[0])
np.random.shuffle(batch)
batch = batch.reshape((-1, batch_size))
batch.sort()

# normalize data from [0,1] to [-1,1] and reshape to (batch_size, channels, res_x, rez_y)
in_tmp = scale_img(test_in[batch[0],...]).reshape((batch_size, 1, img_res, img_res))
out_tmp = scale_img(test_out[batch[0],...]).reshape((batch_size, 1, img_res, img_res))

with torch.no_grad():

    # convert the batch to a tensor and load it onto the GPU
    in_imgs = torch.from_numpy(in_tmp).float().to(device)
    ref_imgs = torch.from_numpy(out_tmp).float().to(device)

    # predict
    net_imgs = net(in_imgs)

    # show
    show_imgs(in_imgs, net_imgs, ref_imgs)
