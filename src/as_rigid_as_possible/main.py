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
from os import listdir
from os.path import isdir, isfile, join
import sys
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
import progressbar as pb
import os
from models import *
import pickle
import time
import gc

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-epoch', type=int, default=110, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--num-updates', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--number-edges', type=int, default=8, metavar='N',
                    help='minimum number of edges per vertex (default: 8)')
parser.add_argument('--coarsening-levels', type=int, default=4, metavar='N',
                    help='number of coarsened graphs. (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default="lap",
                    help='lap | dirac | avg | mlp')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def read_data():
    mypath = "as_rigid_as_possible/data_plus"
    files = sorted([f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith("npy"))])

    print("Loading the dataset")
    pbar = pb.ProgressBar()

    sequences = []

    def load_file(seqname):
        sequence = np.load(open(mypath + "/" + seqname, 'rb'))
        new_sequence = []
        for i, frame in enumerate(sequence):
            frame['V'] = torch.from_numpy(frame['V'])
            frame['F'] = torch.from_numpy(frame['F'])
            if i < 10:
                frame['L'] = utils.sp_sparse_to_pt_sparse(frame['L'])

                if args.model == "dir":
                    frame['Di'] = utils.sp_sparse_to_pt_sparse(frame['Di'])
                    frame['DiA'] = utils.sp_sparse_to_pt_sparse(frame['DiA'])
                else:
                    frame['Di'] = None
                    frame['DiA'] = None
            new_sequence.append(frame)

        return new_sequence

    for seqname in pbar(files):
        sequences.append(load_file(seqname))

        if len(sequences) % 100 == 0:
            gc.collect()
        
    return sequences

sequences = read_data()

test_ind = 0

def sample_batch(sequences, is_training, is_fixed=False):
    global test_ind
    indices = []
    offsets = []

    input_frames = 2
    output_frames = 40

    for b in range(args.batch_size):
        if is_training:
            test_ind = 0
            ind = np.random.randint(0, len(sequences) // 10 * 8)
            offsets.append(np.random.randint(0, len(sequences[ind]) - input_frames - output_frames))
        elif not is_fixed:
            ind = len(sequences) // 10 * 8 + test_ind
            offsets.append(test_ind % (len(sequences[ind]) - input_frames - output_frames))
            test_ind += 1
        elif is_fixed:
            ind = len(sequences) // 10 * 8 + b
            offsets.append(b % (len(sequences[ind]) - input_frames - output_frames))

        sample_batch.num_vertices = max(sample_batch.num_vertices, sequences[ind][0]['V'].size(0))
        sample_batch.num_faces = max(sample_batch.num_faces, sequences[ind][0]['F'].size(0))

        indices.append(ind)

    inputs = torch.zeros(args.batch_size, sample_batch.num_vertices, 3 * input_frames)
    targets = torch.zeros(args.batch_size, sample_batch.num_vertices, 3 * output_frames)
    mask = torch.zeros(args.batch_size, sample_batch.num_vertices, 1)
    faces = torch.zeros(args.batch_size, sample_batch.num_faces, 3).long()
    laplacian = []

    Di = []
    DiA = []

    for b, (ind, offset) in enumerate(zip(indices, offsets)):
        #offset = 0
        num_vertices = sequences[ind][0]['V'].size(0)
        num_faces = sequences[ind][0]['F'].size(0)

        for i in range(input_frames):
            inputs[b, :num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset]['V']

        for i in range(output_frames):
            targets[b, :num_vertices, 3*i:3*(i+1)] = sequences[ind][i + offset + input_frames]['V']

        mask[b, :num_vertices] = 1
        faces[b, :num_faces] = sequences[ind][0]['F']

        L = sequences[ind][offset + input_frames - 1]['L']
        laplacian.append(L)

        if args.model == "dir":
            Di.append(sequences[ind][offset + input_frames - 1]['Di'])
            DiA.append(sequences[ind][offset + input_frames - 1]['DiA'])

    laplacian = utils.sparse_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices)

    if args.model == "dir":
        Di = utils.sparse_cat(Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices)
        DiA = utils.sparse_cat(DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces)

    if args.cuda:
        if args.model == "dir":
            return Variable(inputs).cuda(), Variable(targets).cuda(), Variable(mask).cuda(), Variable(laplacian).cuda(), Variable(Di).cuda(), Variable(DiA).cuda(), faces
        else:
            return Variable(inputs).cuda(), Variable(targets).cuda(), Variable(mask).cuda(), Variable(laplacian).cuda(), None, None, faces
    else:
        return Variable(inputs), Variable(targets), Variable(mask), Variable(laplacian), Variable(Di), Variable(DiA), faces

sample_batch.num_vertices = 0
sample_batch.num_faces = 0

if args.model == "lap":
    model = Model()
elif args.model == "avg":
    model = AvgModel()
elif args.model == "mlp":
    model = MlpModel()
else:
    model = DirModel()

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))

if args.cuda:
    model.cuda()

early_optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
late_optimizer = optim.SGD(model.parameters(), 1e-3, weight_decay=1e-5, momentum=0.9)

def main():
    for epoch in range(args.num_epoch):
        #torch.save(model, 'models/{}_conv.pt'.format(args.model))

        pbar = pb.ProgressBar()
        model.train()
        loss_value = 0
        # Train
        for j in pbar(range(args.num_updates)):
            inputs, targets, mask, laplacian, Di, DiA, faces = sample_batch(sequences, True)
      
            if args.model in ["lap", "avg", "mlp"]:
                outputs = model(laplacian, mask, inputs)
            else:
                outputs = model(Di, DiA, mask, inputs)
       
            outputs = outputs * mask.expand_as(outputs)
            loss = F.smooth_l1_loss(outputs, targets, size_average=False) / args.batch_size
      
            early_optimizer.zero_grad()
            loss.backward()
            early_optimizer.step()
 
            loss_value += loss.data[0]

        print("Train epoch {}, loss {}".format(
            epoch, loss_value / args.num_updates))

        if epoch > 50 and epoch % 10 == 0:
            for param_group in early_optimizer.param_groups:
                param_group['lr'] *= 0.5

        loss_value = 0
        pbar = pb.ProgressBar()

        # Evaluate
        test_trials = len(sequences) // 5 // args.batch_size
        for j in pbar(range(test_trials)):
            inputs, targets, mask, laplacian, Di, DiA, faces = sample_batch(sequences, False)

            if args.model in ["lap", "avg", "mlp"]:
                outputs = model(laplacian, mask, inputs)
            else:
                outputs = model(Di, DiA, mask, inputs)

            outputs = outputs * mask.expand_as(outputs)
            loss = F.smooth_l1_loss(outputs, targets, size_average=False) / args.batch_size
            loss.backward() # because of a problem with caching

            loss_value += loss.data[0]

        print("Test epoch {}, loss {}".format(epoch, loss_value / test_trials))

        inputs, targets, mask, laplacian, Di, DiA, faces = sample_batch(sequences, False, is_fixed=True)

        if args.model in ["lap", "avg", "mlp"]:
            outputs = model(laplacian, mask, inputs)
        else:
            outputs = model(Di, DiA, mask, inputs)
        outputs = outputs * mask.expand_as(outputs)

        results_path = 'as_rigid_as_possible/results_{}'.format(args.model)
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        for k in range(args.batch_size):
            for t in range(inputs.size(2) // 3):
                mesh.save_as_obj(
                    results_path + '/samples_epoch_%03d_%03d_0curr.obj' % (k, t), inputs.data[k,:, 3*t:3*(t+1)].cpu(), faces[k].cpu())

            for t in range(targets.size(2) // 3):
                mesh.save_as_obj(
                    results_path + '/samples_epoch_%03d_%03d_2targ.obj' % (k, t), targets.data[k,:, 3*t:3*(t+1)].cpu(), faces[k].cpu())

                mesh.save_as_obj(
                    results_path + '/samples_epoch_%03d_%03d_1pred.obj' % (k, t), outputs.data[k,:, 3*t:3*(t+1)].cpu(), faces[k].cpu())

if __name__ == "__main__":
    main()
