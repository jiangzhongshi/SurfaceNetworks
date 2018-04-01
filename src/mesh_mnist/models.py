'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

import torch
import torch.nn as nn
import sys
import math
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 64, batch_norm=None)

        for i in range(5):
            module = utils.LapResNet2(64)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(64, 64, batch_norm="pre")

        self.fc1 = nn.Linear(64, 10)

    def forward(self, inputs, L, mask):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(5):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        x = utils.global_average(x, mask).squeeze()
        x = F.dropout(x, training=self.training)

        x = self.fc1(x)
        return F.log_softmax(x)

class AvgModel(nn.Module):

    def __init__(self):
        super(AvgModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 64, batch_norm=None)

        for i in range(5):
            module = utils.AvgResNet2(64)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(64, 64, batch_norm="pre")

        self.fc1 = nn.Linear(64, 10)

    def forward(self, inputs, L, mask):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(5):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        x = utils.global_average(x, mask).squeeze()
        x = F.dropout(x, training=self.training)

        x = self.fc1(x)
        return F.log_softmax(x)


class MlpModel(nn.Module):

    def __init__(self):
        super(MlpModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 64, batch_norm=None)

        for i in range(5):
            module = utils.MlpResNet2(64)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(64, 64, batch_norm="pre")

        self.fc1 = nn.Linear(64, 10)

    def forward(self, inputs, L, mask):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(5):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        x = utils.global_average(x, mask).squeeze()
        x = F.dropout(x, training=self.training)

        x = self.fc1(x)
        return F.log_softmax(x)

class DirModel(nn.Module):

    def __init__(self):
        super(DirModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 64, batch_norm=None)

        for i in range(5):
            module = utils.DirResNet2(64)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(64, 64, batch_norm="pre")

        self.fc1 = nn.Linear(64, 10)

    def forward(self, inputs, Di, DiA, mask):
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        num_faces = DiA.size(2) // 4

        f = Variable(torch.zeros(batch_size, num_faces, 64))
        if v.is_cuda:
            f = f.cuda()

        for i in range(5):
            v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)

        v = F.elu(v)
        v = self.bn_conv2(v)
        v = F.elu(v)

        v = utils.global_average(v, mask).squeeze()
        v = F.dropout(v, training=self.training)

        v = self.fc1(v)
        return F.log_softmax(v)
