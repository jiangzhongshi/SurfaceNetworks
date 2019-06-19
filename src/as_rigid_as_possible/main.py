'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>

This Source Code Form is subject to the terms of the Mozilla Public License
v. 2.0. If a copy of the MPL was not distributed with this file, You can
obtain one at http://mozilla.org/MPL/2.0/.
'''

from __future__ import absolute_import

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import numpy as np
import scipy as sp
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from models import *
import pickle
import time
import gc
import tqdm
import glob

# Training settings
parser = argparse.ArgumentParser(description='As Rigid As Possible')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-epoch', type=int, default=110, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--num-updates', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default="lap",
                    help='lap | dir | avg | mlp')
parser.add_argument('--dense', action='store_true', default=False)
parser.add_argument('--adj', action='store_true', default=False)
parser.add_argument('--first100', action='store_true', default=False)
parser.add_argument('--id', default='test',
                    help='result identifier')
parser.add_argument('--layer', type=int, default=15)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def load_file(seqname):
    sequence = np.load(seqname, encoding='latin1')
    new_sequence = []
    for i, frame in enumerate(sequence):
        frame['V'] = torch.from_numpy(frame['V'])
        frame['F'] = torch.from_numpy(frame['F'])
        if i < 10:
            #frame['L'] = utils.sp_sparse_to_pt_sparse(frame['L'])
            if args.model == "dir":
                if not args.dense:
                    frame['Di'] = utils.sp_sparse_to_pt_sparse(frame['Di'])
                    frame['DiA'] = utils.sp_sparse_to_pt_sparse(frame['DiA'])
            else:
                frame['Di'] = None
                frame['DiA'] = None
        new_sequence.append(frame)

    return new_sequence

def read_data():
    mypath = "as_rigid_as_possible/data_plus"
    files = sorted(glob.glob(mypath+'/*.npy'))
    if args.first100:
        files = files[:100]

    print("Loading the dataset")

    sequences = []
    for seqname in tqdm.tqdm(files):
        sequences.append(load_file(seqname))
        if len(sequences) % 100 == 0:
            gc.collect()

    return sequences

sequences = read_data()

test_ind = 0

def sample_batch(sequences, is_training, is_fixed=False):
    global test_ind
    indices = []
    sample_batch.num_vertices = 0
    sample_batch.num_faces = 0
    offsets = []
    dense_L = args.dense

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
    if args.dense:
        if args.model == 'dir':
            Di = torch.zeros(args.batch_size, 4* sample_batch.num_faces,4* sample_batch.num_vertices)
            DiA = torch.zeros(args.batch_size, 4*sample_batch.num_vertices, 4*sample_batch.num_faces)
        else:
            laplacian = torch.zeros(args.batch_size, sample_batch.num_vertices, sample_batch.num_vertices)
    else:
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
        if args.model == "dir":
            di, dia = sequences[ind][offset + input_frames - 1]['Di'],sequences[ind][offset + input_frames - 1]['DiA']
            if args.dense:
                Di[b, :(4*num_faces), :(4*num_vertices)] = torch.from_numpy(di.todense())
                DiA[b, :(4*num_vertices), :(4*num_faces)] =torch.from_numpy(dia.todense())
            else:
                Di.append(sequences[ind][offset + input_frames - 1]['Di'])
                DiA.append(sequences[ind][offset + input_frames - 1]['DiA'])
        else:
            L = sequences[ind][offset + input_frames - 1]['L']
            if args.dense:
                laplacian[b, :num_vertices, :num_vertices] = torch.from_numpy(L.todense())
            else:
                laplacian.append(utils.sp_sparse_to_pt_sparse(L))



    if not args.dense:
        if args.model == "dir":
            Di = utils.sparse_diag_cat(Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices)
            DiA = utils.sparse_diag_cat(DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces)
        else:
            laplacian = utils.sparse_diag_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices)

    if args.cuda:
        if args.model == "dir":
            return (inputs).cuda(), (targets).cuda(), (mask).cuda(), None, (Di).cuda(), (DiA).cuda(), faces
        else:
            return (inputs).cuda(), (targets).cuda(), (mask).cuda(), (laplacian).cuda(), None, None, faces
    else:
        return (inputs), (targets), (mask), (laplacian), (Di), (DiA), faces


if args.model == "lap":
    model = Model(args.layer, args.dense)
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

        model.train()
        loss_value = 0
        # Train
        for j in tqdm.tqdm(range(args.num_updates)):
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

            loss_value += loss.item()

        print("Train epoch {}, loss {}".format(
            epoch, loss_value / args.num_updates))

        if epoch > 50 and epoch % 10 == 0:
            for param_group in early_optimizer.param_groups:
                param_group['lr'] *= 0.5

        loss_value = 0

        # Evaluate
        with torch.no_grad():
            test_trials = len(sequences) // 5 // args.batch_size
            for j in tqdm.tqdm(range(test_trials)):
                inputs, targets, mask, laplacian, Di, DiA, faces = sample_batch(sequences, False)

                if args.model in ["lap", "avg", "mlp"]:
                    outputs = model(laplacian, mask, inputs)
                else:
                    outputs = model(Di, DiA, mask, inputs)

                outputs = outputs * mask.expand_as(outputs)
                loss = F.smooth_l1_loss(outputs, targets, size_average=False) / args.batch_size
                loss_value += loss.item()

            print("Test epoch {}, loss {}".format(epoch, loss_value / test_trials))

            inputs, targets, mask, laplacian, Di, DiA, faces = sample_batch(sequences, False, is_fixed=True)

            if args.model in ["lap", "avg", "mlp"]:
                outputs = model(laplacian, mask, inputs)
            else:
                outputs = model(Di, DiA, mask, inputs)
            outputs = outputs * mask.expand_as(outputs)

        if epoch % 10 == 9:
            torch.save(model.state_dict(), f'pts/{args.id}_{args.layer}_{args.model}.pts')
        torch.save(model.state_dict(), f'pts/{args.id}_{args.layer}_{args.model}.pts')
if __name__ == "__main__":
    main()
