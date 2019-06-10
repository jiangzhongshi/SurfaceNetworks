'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

from __future__ import absolute_import

import torch
from plyfile import PlyData, PlyElement
import os
from os import listdir
from os.path import isdir, isfile, join
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import numpy as np
import scipy as sp
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import os
import pickle
import time
from models import *
import gc
import torch.backends.cudnn as cudnn
cudnn.benchmark = False

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-epoch', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default="lap",
                    help='lap | avg | mlp | dirac')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


print("Load data")
train_data = np.load(open('mesh_mnist/data/train_plus.np', 'rb'))
test_data = np.load(open('mesh_mnist/data/test_plus.np', 'rb'))

def convert(sample):
    for key in sample.keys():
        class_name = str(type(sample[key]))
        if class_name.find('sparse') > 0:
            sample[key] = utils.sp_sparse_to_pt_sparse(sample[key])
        elif class_name.find('numpy') > 0:
            sample[key] = torch.from_numpy(sample[key])


print("Preprocess Dataset")

for i in tqdm.tqdm(range(len(train_data))):
    convert(train_data[i])


for i in tqdm.tqdm(range(len(test_data))):
    convert(test_data[i])

inputs = torch.zeros(1, 1, 3)
targets = torch.zeros(1).long()
mask = torch.zeros(1, 1, 1)

def sample_batch(samples, is_training):
    indices = []
    for b in range(args.batch_size):
        ind = np.random.randint(0, len(samples))
        sample_batch.num_vertices = max(sample_batch.num_vertices, samples[ind]['V'].size(0))
        sample_batch.num_faces = max(sample_batch.num_faces, samples[ind]['F'].size(0))
        indices.append(ind)

    inputs.resize_(args.batch_size, sample_batch.num_vertices, 3)
    inputs.fill_(0)
    targets.resize_(args.batch_size)
    mask.resize_(args.batch_size, sample_batch.num_vertices, 1)
    mask.fill_(0)
    laplacian = []

    Di = []
    DiA = []

    for b, ind in enumerate(indices):
        num_vertices = samples[ind]['V'].size(0)
        num_faces = samples[ind]['F'].size(0)

        inputs[b, :num_vertices] = samples[ind]['V']
        targets[b] = samples[ind]['label']
        mask[b, :num_vertices] = 1

        laplacian.append(samples[ind]['L'])
        Di.append(samples[ind]['Di'])
        DiA.append(samples[ind]['DiA'])

    laplacian = utils.sparse_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices)
    Di = utils.sparse_cat(Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices)
    DiA = utils.sparse_cat(DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces)

    volatile = False
    if args.cuda:
        return Variable(inputs, volatile=volatile).cuda(), Variable(targets, volatile=volatile).cuda(), Variable(mask, volatile=volatile).cuda(), Variable(laplacian, volatile=volatile).cuda(), Variable(Di, volatile=volatile).cuda(), Variable(DiA, volatile=volatile).cuda()
    else:
        return Variable(inputs, volatile=volatile), Variable(targets, volatile=volatile), Variable(mask, volatile=volatile), Variable(laplacian, volatile=volatile), Variable(Di, volatile=volatile), Variable(DiA, volatile=volatile)

sample_batch.num_vertices = 0
sample_batch.num_faces = 0

if args.model == "lap":
    model = Model()
elif args.model == "mlp":
    model = MlpModel()
elif args.model == "avg":
    model = AvgModel()
else:
    model = DirModel()

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))

if args.cuda:
    model.cuda()

early_optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)


def main():
    for epoch in range(args.num_epoch):

        
        model.train()
        loss_value = 0.0
        correct = 0.0

        # Train
        for j in tqdm.tqdm(range(len(train_data) // args.batch_size)):
            inputs, targets, mask, laplacian, Di, DiA = sample_batch(train_data, is_training=True)

            if args.model in ["lap", "avg", "mlp"]:
                outputs = model(inputs, laplacian, mask)
            else:
                outputs = model(inputs, Di, DiA, mask)

            loss = F.nll_loss(outputs, targets)

            early_optimizer.zero_grad()
            loss.backward()
            early_optimizer.step()

            loss_value += loss.data[0]
            pred = outputs.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(targets.data).cpu().sum()
        gc.collect()

        print("Train epoch {}, loss {}, acc {}".format(epoch,
                loss_value / (len(train_data) // args.batch_size),
                correct / (len(train_data) // args.batch_size * args.batch_size)))

        if epoch > 20 and epoch % 10 == 0:
            for param_group in early_optimizer.param_groups:
                param_group['lr'] *= 0.5

        model.eval()

        for module in model.modules():
            if module.__class__.__name__.find("BatchNorm") > -1:
                module.train()
                # BatchNorm for some reasons is not stable in eval

        loss_value = 0.0
        correct = 0.0
        

        # Evaluate
        for j in tqdm.tqdm(range(len(test_data) // args.batch_size)):
            inputs, targets, mask, laplacian, Di, DiA = sample_batch(test_data, is_training=False)

            if args.model in ["lap", "avg", "mlp"]:
                outputs = model(inputs, laplacian, mask)
            else:
                outputs = model(inputs, Di, DiA, mask)

            loss = F.nll_loss(outputs, targets)

            loss.backward()

            loss_value += loss.data[0]
            pred = outputs.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(targets.data).cpu().sum()
            gc.collect()

        print("Test epoch {}, loss {}, acc {}".format(epoch,
                        loss_value / (len(test_data) /args.batch_size),
                        correct / (len(test_data) // args.batch_size * args.batch_size)))


if __name__ == "__main__":
    main()
