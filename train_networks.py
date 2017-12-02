#/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.modules.loss import CrossEntropyLoss

from networks import CovNet


def data_loader(file_name, val_pct=0.2):

    with h5.File(file_name, 'r') as data_file:
        total_n_pics = data_file['labels'].__len__()
        label_names = data_file['label_names'][:]
        data = data_file['standardized_pics'][:]
        labels = data_file['labels'][:]

    total_idx = np.random.permutation(range(total_n_pics))

    train_idx = np.sort(total_idx[int(val_pct * total_n_pics):]).tolist()
    train_data = data[train_idx]
    train_labels = labels[train_idx]

    val_idx = np.sort(total_idx[:int(val_pct * total_n_pics)]).tolist()
    val_data = data[val_idx]
    val_labels = labels[val_idx]

    train_data_loader = DataLoader(
        dataset=TensorDataset(
            data_tensor=torch.from_numpy(train_data),
            target_tensor=torch.from_numpy(train_labels)
        ),
        shuffle=True,

    )
    val_data_loader = DataLoader(
        dataset=TensorDataset(
            data_tensor=torch.from_numpy(val_data),
            target_tensor=torch.from_numpy(val_labels)
        ),
        shuffle=True,

    )

    return train_data_loader, val_data_loader, label_names


def train(params):

    train_data_loader, val_data_loader, label_names = data_loader(params['data_file'])
    train_data_loader.batch_size = params['batch_size']
    val_data_loader.batch_size = params['batch_size']

    if params['existing_model'] is None:
        net = (
            CovNet(
                n_channel_multiple=params['n_channel_multiple'],
                prob=params['drop_p']
            ).cuda(params['gpu_idx'])
        )
    else:
        net = torch.load(params['existing_model'])

    optimizer = Adam(
        net.parameters(),
        lr=params['lr']
    )
    criterion = CrossEntropyLoss()

    train_loss_recorder = []
    val_loss_recorder = []

    train_accuracy_recorder = []
    val_accuracy_recorder = []

    for epoch_idx in xrange(params['n_epoch']):

        print(params)

        running_loss = 0.0
        running_correct = 0
        batch_count = 0
        for batch in train_data_loader:
            optimizer.zero_grad()
            data = Variable(batch[0].cuda(params['gpu_idx']))
            target = Variable(batch[1].cuda(params['gpu_idx']))
            output = net(data).view(-1, params['n_class'])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            running_correct += output.max(1)[1].eq(target).sum().data[0]
            batch_count += 1
            if batch_count % params['show_batch_frequency'] == params['show_batch_frequency'] - 1:
                accuracy = output.max(1)[1].eq(target).sum().data[0] / float(params['batch_size'])
                print([batch_count, loss.data[0] / params['batch_size'], accuracy])
        train_loss = running_loss / (batch_count * params['batch_size'])
        train_accuracy = running_correct / float(batch_count * params['batch_size'])
        train_loss_recorder.append(train_loss)
        train_accuracy_recorder.append(train_accuracy)
        print([epoch_idx, train_loss, train_accuracy])

        running_loss = 0.0
        running_correct = 0
        batch_count = 0
        for batch in val_data_loader:
            data = Variable(batch[0].cuda(params['gpu_idx']))
            target = Variable(batch[1].cuda(params['gpu_idx']))
            output = net(data).view(-1, params['n_class'])
            loss = criterion(output, target)
            running_loss += loss.data[0]
            running_correct += output.max(1)[1].eq(target).sum().data[0]
            batch_count += 1
        val_loss = running_loss / (batch_count * params['batch_size'])
        val_accuracy = running_correct / float(batch_count * params['batch_size'])
        val_loss_recorder.append(val_loss)
        val_accuracy_recorder.append(val_accuracy)
        print([epoch_idx, val_loss, val_accuracy])

        if epoch_idx % params['save_frequency'] == params['save_frequency'] - 1:

            current_time = str(int(time()))

            # save model
            if not os.path.exists(os.path.join(os.getcwd(), 'saved')):
                os.makedirs(os.path.join(os.getcwd(), 'saved'))
            torch.save(
                net,
                os.path.join(
                    os.getcwd(),
                    'saved',
                    (
                        current_time
                        +
                        '_'
                        +
                        ('%0.2f' % (val_accuracy * 100.0)).replace('.', '_')
                        +
                        '.model'
                    )
                )
            )

            # save pic
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(train_accuracy_recorder, 'k-x')
            ax.plot(val_accuracy_recorder, 'r-x')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Top 1 Accuracy')
            ax.legend(['Train Accuracy', 'Val Accuracy'], loc=0)

            if not os.path.exists(os.path.join(os.getcwd(), 'pics')):
                os.makedirs(os.path.join(os.getcwd(), 'pics'))
            fig.savefig(
                os.path.join(
                    os.getcwd(),
                    'pics',
                    (
                        current_time
                        +
                        '_'
                        +
                        ('%0.2f' % (val_accuracy * 100.0)).replace('.', '_')
                        +
                        '.png'
                    )
                )
            )


if __name__ == '__main__':

    params = {

        'n_class': 62,

        'drop_p': 0.1,
        'n_channel_multiple': 8,

        'lr': 1e-6,
        'batch_size': 200,
        'n_epoch': 151,
        'gpu_idx': 0,

        'save_frequency': 10,
        'show_batch_frequency': 100,

        'existing_model': None,
        'data_file': './data/hwc_32_x_32.h5'

    }

    train(params)
