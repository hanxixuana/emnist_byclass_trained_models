#!/usr/bin/env python

import torch
from torch import nn


class CovNet(nn.Module):

    def __init__(self, n_channel_multiple=8, prob=0.1):
        super(CovNet, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 4*n_channel_multiple, kernel_size=7, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=1),

            nn.Conv2d(4*n_channel_multiple, 16*n_channel_multiple, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(16*n_channel_multiple, 32*n_channel_multiple, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(32 * n_channel_multiple, 64 * n_channel_multiple, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(64*n_channel_multiple, 128*n_channel_multiple, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=prob),
            nn.Conv2d(128*n_channel_multiple, 64*n_channel_multiple, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(64*n_channel_multiple, 32*n_channel_multiple, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32*n_channel_multiple, 62, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
