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

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None)

        for i in range(15):
            if i % 2 == 0:
                module = utils.LapResNet2(128)
            else:
                module = utils.AvgResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre")

    def forward(self, L, mask, inputs):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(15):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)

class AvgModel(nn.Module):

    def __init__(self):
        super(AvgModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None)

        for i in range(15):
            module = utils.AvgResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre")

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(15):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)


class MlpModel(nn.Module):

    def __init__(self):
        super(MlpModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None)

        for i in range(15):
            module = utils.MlpResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.bn = utils.GraphBatchNorm(128)
        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm=None)

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(15):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = self.bn(x)
        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)


class DirModel(nn.Module):

    def __init__(self):
        super(DirModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(6, 128, batch_norm=None)

        for i in range(15):
            if i % 2 == 0:
                module = utils.DirResNet2(128)
            else:
                module = utils.AvgResNet2(128)

            self.add_module("rn{}".format(i), module)

        self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre")


    def forward(self, Di, DiA, mask, inputs):
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        num_faces = DiA.size(2) // 4

        f = Variable(torch.zeros(batch_size, num_faces, 128))
        if v.is_cuda:
            f = f.cuda()

        for i in range(15):
            if i % 2 == 0:
                v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
            else:
                v = self._modules['rn{}'.format(i)](None, mask, v)

        x = v
        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)
