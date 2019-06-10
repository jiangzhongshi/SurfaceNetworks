'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>

This Source Code Form is subject to the terms of the Mozilla Public License
v. 2.0. If a copy of the MPL was not distributed with this file, You can
obtain one at http://mozilla.org/MPL/2.0/.
'''

import torch
import os
import datetime
from os import listdir
from os.path import isdir, isfile, join
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
import utils.mesh as mesh
import utils.utils_pt as utils
import numpy as np
import scipy as sp
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
from models import *
import pickle
import time
import gc
import itertools
import random
import subprocess
# Training settings
parser = argparse.ArgumentParser(description='SurfaceNN dense correspondence example')

parser.add_argument('--batch-size', type=int, default=1, metavar='N')
parser.add_argument('--datapath', default="/dev/shm/FAUST_npz/train_FAUST_npz/", help='datapath')
parser.add_argument('--deser-option', default='auto', choices= ['auto', 'no', 'force'], help='deser-option')
parser.add_argument('--deser-path', default=None, help='deserialization path')

parser.add_argument('--layer', type = int, default=15)
parser.add_argument('--loss', default='dcel')
parser.add_argument('--lr', default='1e-3')
parser.add_argument('--model', default="lap", help='lap | dir | avg | mlp | amp')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--no-pre-load', action='store_true', default=False)
parser.add_argument('--num-epoch', type=int, default=110, metavar='N')
parser.add_argument('--num-updates', type=int, default=100, metavar='N')
parser.add_argument('--result-prefix', default='serious')

parser.add_argument('--off-bn', action='store_true', default=False)
parser.add_argument('--xz-rotate', action='store_true', default=False)
parser.add_argument('--xy-rotate', action='store_true', default=False)
parser.add_argument('--complete-test', action='store_true', default=False)
parser.add_argument('--full-train', action='store_true', default=False)

def time_string():
    time = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(-1, 72000))).strftime("%b-%d-%I:%M%p")
    return f'NewYork {time}'
from scipy.sparse.linalg import norm as spnorm
def read_data(seqname, args):
    with np.load(seqname) as sequence:
        new_sequence = []
        frame = {}
        frame['V'] = torch.from_numpy(sequence['V'].astype('f'))

        frame['F'] = torch.from_numpy(sequence['F'])

        if 'amp' in args.model:
            frame['L'] = []
            L = sequence['L'].item().astype('f').tocsr()
            idp = L.indptr
            Dsq = sp.sparse.diags(1/np.sqrt(idp[1:] - idp[:-1] - 1)).astype('f')
            L = Dsq.dot(L).dot(Dsq).astype('f')
            frame['L'].append(utils.sp_sparse_to_pt_sparse(L).coalesce())

            for i in range(2):
                L = Dsq.dot(L).dot(Dsq).astype('f')
                L = L.dot(L).tocsr()
                frame['L'].append(utils.sp_sparse_to_pt_sparse(L).coalesce())

        elif 'lap' in args.model:
            L = sequence['L'].item().astype('f').tocsr()
            frame['L'] = utils.sp_sparse_to_pt_sparse(L).coalesce()
        else:
            frame['L'] = None

        if 'dir' in args.model:
            frame['Di'] = utils.sp_sparse_to_pt_sparse(sequence['D'].item().astype('f')).coalesce()
            frame['DiA'] = utils.sp_sparse_to_pt_sparse(sequence['DA'].item().astype('f')).coalesce()
        else:
            frame['Di'] = None
            frame['DiA'] = None

        frame['label'] = torch.from_numpy(sequence['label'])
        frame['label_inv'] = torch.from_numpy(sequence['label_inv'])
        frame['G'] = torch.from_numpy(sequence['dist_mat'].astype('f'))

    return frame

def sample_batch(sequences, is_training, args):

    indices = []
    offsets = []

    input_frames = 1
    output_frames = 40
    gc.collect()
    for b in range(args.batch_size):
        if is_training:
            sample_batch.test_ind = 0
            ind = np.random.randint(0, len(sequences) // 10 * 8)
            if args.full_train:
                ind = np.random.randint(0, len(sequences))
            offsets.append(0)
        else:
            ind = sample_batch.test_ind
            offsets.append(0)

        sequence_ind = (sequences[ind])
        if type(sequence_ind) == str:
            sequence_ind = read_data(sequence_ind, args)

        sample_batch.num_vertices = max(sample_batch.num_vertices, sequence_ind['V'].size(0))
        sample_batch.num_faces = max(sample_batch.num_faces, sequence_ind['F'].size(0))

        indices.append(ind)

    inputs = torch.zeros(args.batch_size, sample_batch.num_vertices, 3 * input_frames)
    #targets = (torch.zeros(args.batch_size, sample_batch.num_vertices, sample_batch.num_vertices).cuda(), torch.zeros(args.batch_size, sample_batch.num_vertices).cuda(), torch.zeros(args.batch_size, sample_batch.num_vertices).cuda())
    mask = torch.zeros(args.batch_size, sample_batch.num_vertices, 1)
    faces = torch.zeros(args.batch_size, sample_batch.num_faces, 3).long()
    if 'amp' in args.model:
        laplacian = [[] for i in range(args.layer)]
    laplacian = []
    targets = [None]*args.batch_size

    Di = []
    DiA = []

    for b, (ind, offset) in enumerate(zip(indices, offsets)):
        num_vertices = sequence_ind['V'].size(0)
        num_faces = sequence_ind['F'].size(0)

        inputV = sequence_ind['V']
        if args.xz_rotate:
            rotate_matrix_xz = lambda t: torch.Tensor([[np.cos(t), 0, np.sin(t)], [0,1,0],[-np.sin(t), 0, np.cos(t)]])
            inputV = torch.matmul(inputV, rotate_matrix_xz(random.random()*2*np.pi))
        if args.xy_rotate:
            rotate_matrix_xy = lambda t: torch.Tensor([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0,0,1]])
            inputV = torch.matmul(inputV, rotate_matrix_xy(random.random()*2*np.pi))

        inputs[b, :num_vertices, :3] = inputV

        targets[b] = (sequence_ind['G'].cuda(),
                      sequence_ind['label'].cuda(),
                      sequence_ind['label_inv'].cuda())

        mask[b, :num_vertices] = 1
        faces[b, :num_faces] = sequence_ind['F']

        if 'amp' in args.model:
            L = sequence_ind['L']
            laplacian.append(L)
        elif 'lap' in args.model:
            L = sequence_ind['L']
            laplacian.append(L)

        if 'dir' in args.model:
            Di.append(sequence_ind['Di'])
            DiA.append(sequence_ind['DiA'])
    if 'amp' in args.model:
        laplacian = [utils.sparse_cat(lap, sample_batch.num_vertices, sample_batch.num_vertices).coalesce() for lap in map(list, zip(*laplacian))]
    elif 'lap' in args.model:
        laplacian = utils.sparse_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices).coalesce()

    Operator = None
    if 'dir' in args.model:
        Di = utils.sparse_cat(Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices).coalesce()
        DiA = utils.sparse_cat(DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces).coalesce()
        Operator = ((Di).cuda(), (DiA).cuda())
    elif 'amp' in args.model:
        Operator = [(lap).cuda() for lap in laplacian]
    elif 'lap' in args.model:
        Operator = (laplacian).cuda()
    return (inputs).cuda(), (targets), (mask).cuda(), Operator, faces

sample_batch.num_vertices = 7000
sample_batch.num_faces = 0
sample_batch.test_ind = 0

def aggregate_batch_G(outputs, targetX, targetY):
    batch_size = outputs.size(0)
    listG = []
    for i in range(batch_size):
        GA, lA, liA = targetX[i]
        GB, lB, liB = targetY[i]
        NA = lA.size(0)
        NB = lB.size(0)
        G = torch.cuda.FloatTensor(outputs.size(1), outputs.size(2)).zero_()
        G[:NA, :NB] = GA[:, liA[lB]] + GB[liB[lA],:]
        listG.append(G)

    FullG = torch.stack(listG)
    return FullG

def loss_fun_sl1(outputs, targetX, targetY):
    FullG = aggregate_batch_G(outputs, targetX, targetY)
    return F.smooth_l1_loss(outputs, (FullG), size_average=True) / (outputs.size(0))

def loss_fun_cross_entropy(outputs, targetX, targetY):
    batch_size = outputs.size(0)
    loss = (torch.cuda.FloatTensor(1).zero_())
    for i in range(batch_size):
        GA, lA, liA = targetX[i]
        GB, lB, liB = targetY[i]
        NA = lA.size(0)
        NB = lB.size(0)
        GAB = (GA[:, liA[lB]] + GB[liB[lA],:])
        G = F.softmin(GAB, dim = 1)
        loss = loss - torch.dot(G, F.log_softmax(outputs[0,:NA, :NB], dim = 1))
    return loss / batch_size

def loss_fun_delta_cross_entropy(outputs, targetX, targetY):
    batch_size = outputs.size(0)
    listG = []
    loss = (torch.cuda.FloatTensor(1).zero_())
    for i in range(batch_size):
        GA, lA, liA = targetX[i]
        GB, lB, liB = targetY[i]
        NA = lA.size(0)
        NB = lB.size(0)
        _, GAB = torch.min(GA[:, liA[lB]] + GB[liB[lA],:], dim = 1)
        loss = loss + F.cross_entropy(outputs[0, :NA, :NB], (GAB))
    return loss/ batch_size

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        assert False, 'No CUDA'

    result_identifier = args.result_prefix

    def custom_logging(stuff):
        print(f'{result_identifier}::{stuff}', file=sys.stderr) # also to err
        logfile = f'{result_identifier}.log'
        with open(logfile,'a') as fp:
            print(stuff, file=fp)

    sequences = sorted(glob.glob(args.datapath+'/*.npz'))

    model = (SiameseModel(args.model, args.layer))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    custom_logging(f'NewYork {time_string()}')
    custom_logging("Num parameters {}".format(num_params))
    custom_logging(subprocess.check_output('hostname'))

    if args.cuda:
        model.cuda()

    def load_deser(path):
        custom_logging('Continue...')
        model.load_state_dict(torch.load(path))
        if path[-1] != '_':
            shutil.move(path, path+'_')

    if args.deser_option != 'no':
        deser_path = args.deser_path
        if args.deser_path is not None: # auto
            load_deser(deser_path)
        else:
            deser_path = f'pts/{result_identifier}_state.pts'
            if os.path.isfile(deser_path): # auto -> yes
                load_deser(deser_path)

    early_optimizer = optim.Adam(model.parameters(),float(args.lr), weight_decay=1e-5)
    late_optimizer = optim.SGD(model.parameters(), 1e-3, weight_decay=1e-5, momentum=0.9)

    if args.loss == 'sl1':
        loss_fun = loss_fun_sl1
    elif args.loss == 'cel':
        loss_fun = loss_fun_cross_entropy
    elif args.loss == 'dcel':
        loss_fun = loss_fun_delta_cross_entropy

    if not args.no_pre_load:
        sequences = [read_data(n, args) for n in sequences]
    custom_logging("Data Read")

    for epoch in range(args.num_epoch):
        gc.collect()
        model.train()
        loss_value = 0
        # Train
        if args.off_bn:
            for module in model.modules():
                classname = module.__class__.__name__
            if classname.find('BatchNorm') > -1:
                module.eval()

        for j in (range(args.num_updates)):

            inputX, targetX, maskX, OperatorX, facesX = sample_batch(sequences, True, args)
            inputY, targetY, maskY, OperatorY, facesY = sample_batch(sequences, True, args)

            if 'dir' in args.model:
                DiX, DiAX = OperatorX
                DiY, DiAY = OperatorY
                outputs = model([DiX, DiAX, maskX],[DiY, DiAY, maskY], inputX, inputY)
            else:
                outputs = model([OperatorX, maskX],[OperatorY, maskY], inputX, inputY)

            loss = loss_fun(outputs, targetX, targetY)

            early_optimizer.zero_grad()
            loss.backward()
            early_optimizer.step()
            loss_value += loss.item()

            if np.any(np.isnan(outputs.data)):
                assert False, (result_identifier, epoch, j)

        custom_logging("Train epoch {}, loss {}".format(
            epoch, loss_value / args.num_updates))

        loss_value = 0

        testing_pairs = list(itertools.product(range(80,100),repeat=2)) + list(itertools.product(range(80,100),range(0,80)))
        if not args.complete_test:
            testing_pairs = random.choices(testing_pairs, k=20)

        # Evaluate
        for i, j in testing_pairs:
            sample_batch.test_ind = i
            inputX, targetX, maskX, OperatorX, facesX = sample_batch(sequences, False, args)
            sample_batch.test_ind = j
            inputY, targetY, maskY, OperatorY, facesY = sample_batch(sequences, False, args)
            if 'dir' in args.model:
                DiX, DiAX = OperatorX
                DiY, DiAY = OperatorY
                outputs = model([DiX, DiAX, maskX],[DiY, DiAY, maskY], inputX, inputY)
            else:
                outputs = model([OperatorX, maskX],[OperatorY, maskY], inputX, inputY)

            mask = torch.bmm(maskX, maskY.transpose(1,2))
            outputs = outputs * mask.expand_as(outputs)
            loss = loss_fun(outputs, targetX, targetY)
            loss.backward() # because of a problem with caching

            loss_value += loss.item()

        custom_logging("Test epoch {}, loss {}".format(epoch, loss_value / len(testing_pairs)))

        if epoch % 10 == 9:
            torch.save(model.state_dict(),f'pts/{result_identifier}_state.pts')
    custom_logging(f'NewYork {time_string()}')

if __name__ == "__main__":
    main()
