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

class LapEncoder(nn.Module):
    def __init__(self):
        super(LapEncoder, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.num_layers = 5
        for i in range(self.num_layers):
            module = utils.LapResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(128, 128, batch_norm="pre")

        self.fc_mu = nn.Linear(128, 100)
        self.fc_logvar = nn.Linear(128, 100)

    def forward(self, inputs, L, mask):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)

        for i in range(self.num_layers):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        x = utils.global_average(x, mask).squeeze()

        return self.fc_mu(x), self.fc_logvar(x)


class LapDecoder(nn.Module):
    def __init__(self):
        super(LapDecoder, self).__init__()

        self.conv_inputs = utils.GraphConv1x1(3, 128, batch_norm=None)
        self.conv_noise = utils.GraphConv1x1(100, 128, batch_norm=None)

        self.num_layers = 5
        for i in range(self.num_layers):
            module = utils.LapResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(128, 128, batch_norm="pre")

        self.fc_mu = utils.GraphConv1x1(128, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, noise, L, mask):
        batch_size, num_nodes, _ = inputs.size()
        x = self.conv_inputs(inputs) + self.conv_noise(noise)

        for i in range(self.num_layers):
            x = self._modules['rn{}'.format(i)](L, mask, x)

        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        mu = self.fc_mu(x)

        y = self.fc_logvar.expand_as(mu).contiguous()

        return mu + inputs, y

class LapVAE(nn.Module):

    def __init__(self):
        super(LapVAE, self).__init__()

        self.encoder = LapEncoder()
        self.decoder = LapDecoder()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, flat_x, L, flat_L, mask):
        mu, logvar = self.encoder(x, L, mask)

        z = self.reparametrize(mu, logvar)

        z_ = z.unsqueeze(1)
        z_ = z_.repeat(1, flat_x.size(1), 1)

        recog_mu, recog_logvar = self.decoder(flat_x, z_, flat_L, mask)
        return recog_mu, recog_logvar, z, mu, logvar


class DirEncoder(nn.Module):
    def __init__(self):
        super(DirEncoder, self).__init__()

        self.conv1 = utils.GraphConv1x1(3, 128, batch_norm=None)

        self.num_layers = 5
        for i in range(self.num_layers):
            module = utils.DirResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(128, 128, batch_norm="pre")

        self.fc_mu = nn.Linear(128, 100)
        self.fc_logvar = nn.Linear(128, 100)

    def forward(self, inputs, Di, DiA, mask):
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        num_faces = DiA.size(2) // 4

        f = Variable(torch.zeros(batch_size, num_faces, 128))

        if v.is_cuda:
            f = f.cuda()

        for i in range(self.num_layers):
            v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)

        x = v
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        x = utils.global_average(x, mask).squeeze()

        return self.fc_mu(x), self.fc_logvar(x)


class DirDecoder(nn.Module):
    def __init__(self):
        super(DirDecoder, self).__init__()

        self.conv_inputs = utils.GraphConv1x1(3, 128, batch_norm=None)
        self.conv_noise = utils.GraphConv1x1(100, 128, batch_norm=None)

        self.num_layers = 5
        for i in range(self.num_layers):
            module = utils.DirResNet2(128)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = utils.GraphConv1x1(128, 128, batch_norm="pre")

        self.fc_mu = utils.GraphConv1x1(128, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, noise, Di, DiA, mask):
        batch_size, num_nodes, _ = inputs.size()
        v = self.conv_inputs(inputs) + self.conv_noise(noise)

        num_faces = DiA.size(2) // 4

        f = Variable(torch.zeros(batch_size, num_faces, 128))

        if v.is_cuda:
            f = f.cuda()

        for i in range(self.num_layers):
            v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)

        x = v
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        mu = self.fc_mu(x)

        y = self.fc_logvar.expand_as(mu).contiguous()

        return mu + inputs, y

class DirVAE(nn.Module):

    def __init__(self):
        super(DirVAE, self).__init__()

        self.encoder = DirEncoder()
        self.decoder = DirDecoder()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, flat_x, Di, DiA, flat_Di, flat_DiA, mask):
        mu, logvar = self.encoder(x, Di, DiA, mask)

        z = self.reparametrize(mu, logvar)

        z_ = z.unsqueeze(1)
        z_ = z_.repeat(1, flat_x.size(1), 1)

        recog_mu, recog_logvar = self.decoder(flat_x, z_, flat_Di, flat_DiA, mask)
        return recog_mu, recog_logvar, z, mu, logvar
