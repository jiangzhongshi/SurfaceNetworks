'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>

This Source Code Form is subject to the terms of the Mozilla Public License
v. 2.0. If a copy of the MPL was not distributed with this file, You can
obtain one at http://mozilla.org/MPL/2.0/.
'''

import os, sys
sys.path.append(os.path.dirname(__file__))
from cuda.sparse_bmm_func import SparseBMMFunc

import scipy as sp
import numpy as np
import utils.graph as graph
import torch
import torch.nn as nn
import torch.nn.functional as F

def sparse_cat(tensors, size0, size1):
    values = []
    tensor_size = 0
    for i, tensor in enumerate(tensors):
        values.append(tensor._values())
        tensor_size += tensor._nnz()

    indices = torch.LongTensor(3, tensor_size)
    index = 0
    for i, tensor in enumerate(tensors):
        indices[0, index:index+tensor._nnz()] = i
        indices[1:3, index:index+tensor._nnz()].copy_(tensor._indices())

        index += tensor._nnz()

    values = torch.cat(values, 0)

    size = torch.Size((len(tensors), size0, size1))
    return torch.sparse.FloatTensor(indices, values, size).coalesce()

def sparse_diag_cat(tensors, size0, size1):
    assert size0 == size1
    N = size0
    values = []
    for i, tensor in enumerate(tensors):
        values.append(tensor._values())
    indices = []
    index = 0
     # assuming COO
    for i, t in enumerate(tensors):
        indices.append(t._indices()+i*N)
    values = torch.cat(values, 0)
    indices = torch.cat(indices, 1)
    size = torch.Size((len(tensors)*size0, len(tensors)*size1))
    return torch.sparse.FloatTensor(indices, values, size).coalesce()


def sp_sparse_to_pt_sparse(L):
    """
    Converts a scipy matrix into a pytorch one.
    """
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    indices = torch.from_numpy(indices).long()
    L_data = torch.from_numpy(L.data)

    size = torch.Size(L.shape)
    indices = indices.transpose(1, 0)

    L = torch.sparse.FloatTensor(indices, L_data, size)
    return L

def to_dense_batched(x, batch_size):
    x = x.to_dense()
    x = x.unsqueeze(0)
    return x.repeat(batch_size, 1, 1)

class GraphConv1x1(nn.Module):
    def __init__(self, num_inputs, num_outputs, batch_norm=None):
        super(GraphConv1x1, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm

        if self.batch_norm == "pre":
            self.bn = nn.BatchNorm1d(num_inputs)

        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs)

        self.fc = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        batch_size, num_nodes, num_inputs = x.size()
        assert num_inputs == self.num_inputs

        x = x.contiguous()
        x = x.view(-1, self.num_inputs)

        if self.batch_norm == "pre":
            x = self.bn(x)
        x = self.fc(x)
        if self.batch_norm == "post":
            x = self.bn(x)

        x = x.view(batch_size, num_nodes, self.num_outputs)
        return x


class GraphBatchNorm(nn.Module):
    def __init__(self, num_inputs):
        super(GraphBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_inputs)

    def forward(self, x):
        self.bn.train()
        batch_size, num_nodes, num_inputs = x.size()
        x = x.view(batch_size * num_nodes, num_inputs)
        x = self.bn(x)
        x = x.view(batch_size, num_nodes, num_inputs)
        return x

def global_average(x, mask):
    mask = mask.expand_as(x)
    return (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)

class LapResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(LapResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, mask, inputs):
        x = inputs
        x = F.elu(x)

        batch, node, dim = x.size()
        Lx = torch.mm(L, x.view(-1, dim)).view(batch, node, dim)
        xs = [x, Lx]

        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)

        x = F.elu(x)
        Lx = torch.mm(L, x.view(-1, dim)).view(batch, node, dim)
        xs = [x, Lx]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs

class DirResNet2(nn.Module):
    def __init__(self, num_outputs, res_f=False):
        super(DirResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.res_f = res_f

    def forward(self, Di, DiA, v, f):
        batch_size, num_nodes, num_inputs = v.size()
        _, num_faces, _ = f.size()

        x_in, f_in = F.elu(v), F.elu(f)
        x = x_in
        x = x.view(batch_size, num_nodes * 4, num_inputs // 4)
        x = SparseBMMFunc()(Di, x)
        x = x.view(batch_size, num_faces, num_inputs)
        x = torch.cat([f_in, x], 2)
        x = self.bn_fc0(x)
        f_out = x

        x = F.elu(x)
        x = x.view(batch_size, num_faces * 4, num_inputs // 4)
        x = SparseBMMFunc()(DiA, x)
        x = x.view(batch_size, num_nodes, num_inputs)
        x = torch.cat([x_in, x], 2)
        x = self.bn_fc1(x)
        v_out = x

        return v + v_out, f_out

class AvgResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(AvgResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, mask, inputs):
        x = inputs
        x = F.elu(x)

        xs = [x, global_average(x, mask).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)

        x = F.elu(x)
        xs = [x, global_average(x, mask).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs

class MlpResNet2(nn.Module):
    def __init__(self, num_outputs):
        super(MlpResNet2, self).__init__()
        self.num_outputs = num_outputs

        self.bn0 = GraphBatchNorm(num_outputs)
        self.fc0 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None)
        self.bn1 = GraphBatchNorm(num_outputs)
        self.fc1 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None)

    def forward(self, L, mask, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.elu(x)
        x = self.fc0(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.fc1(x)
        return x + inputs
