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
import time
class Model(nn.Module):

    def __init__(self, layer):
        super(Model, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.layer = layer
        for i in range(self.layer):
            if i % 2 == 0:
                module = utils.LapResNet2(128)
            else:
                module = utils.AvgResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre")

    def forward(self, L, mask, inputs):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)

class AmplifyModel(nn.Module):

    def __init__(self, layer):
        super().__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.layer = layer
        for i in range(self.layer):
            if i % 2 == 0:
                module = utils.LapResNet2(128)
            else:
                module = utils.AvgResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre")

    def forward(self, L_sequence, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer):
            if i//2 >= len(L_sequence):
                L = L_sequence[-1]
            else:
                L = L_sequence[i//2]
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)

class AvgModel(nn.Module):

    def __init__(self, layer):
        super(AvgModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.layer = layer
        for i in range(self.layer):
            module = utils.AvgResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm="pre")

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)


class MlpModel(nn.Module):

    def __init__(self, layer):
        super(MlpModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.layer = layer
        for i in range(self.layer):
            module = utils.MlpResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.bn = utils.GraphBatchNorm(128)
        self.conv2 = utils.GraphConv1x1(128, 120, batch_norm=None)

    def forward(self, L, mask, inputs):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.layer):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = self.bn(x)
        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)


class DirModel(nn.Module):

    def __init__(self, layer):
        super(DirModel, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.layer = layer
        for i in range(self.layer):
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

        f = (torch.zeros(batch_size, num_faces, 128))
        if v.is_cuda:
            f = f.cuda()

        for i in range(self.layer):
            if i % 2 == 0:
                v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
            else:
                v = self._modules['rn{}'.format(i)](None, mask, v)

        x = v
        x = F.elu(x)
        x = self.conv2(x)

        return x + inputs[:, :, -3:].repeat(1, 1, 40)

class SiameseModel(nn.Module):
    def __init__(self, model='dirac', layer=15):
        super().__init__()

        if 'dir' in model:
            self.model = DirModel(layer)
        elif 'amp' in model:
            self.model = AmplifyModel(layer)
        elif 'lap' in model:
            self.model = Model(layer)
        elif 'avg' in model:
            self.model = AvgModel(layer)
        elif 'mlp' in model:
            self.model = MlpModel(layer)

    def forward(self, OperationA, OperationB, inputA, inputB):
        FA = self.model(*OperationA, inputA)
        FB = self.model(*OperationB, inputB)

        return torch.bmm(FA, FB.transpose(1,2))
        # Batch * Nodes * Feature
        batch_size = inputA.size(0)
        # ||Ai-Bj||^2 = (normAi^2) + normBj^2 - 2 * AiBj
        AB = -2 * torch.bmm(FA, FB.transpose(1,2))

        sqnormA = torch.sum(FA**2, dim=2)
        sqnormB = torch.sum(FB**2, dim=2)

        # Dimension difference might be handled by broadcasting

        DMat = (AB + sqnormA.view(batch_size,-1,1)) + sqnormB.view(batch_size,1,-1)
        return DMat
