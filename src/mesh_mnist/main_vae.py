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
from os.path import isdir, isfile, join, dirname
import sys
sys.path.append(join(dirname(__file__), '..'))
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
from models_vae import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-epoch', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default="lap",
                    help='lap | dirac')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print("Load data")
train_data = np.load(open('mesh_mnist/data/train_plus.np', 'rb'),encoding='latin1')
test_data = np.load(open('mesh_mnist/data/test_plus.np', 'rb'), encoding='latin1')

def convert(sample):
    for key in list(sample):
        class_name = str(type(sample[key]))
        if class_name.find('sparse') > 0:
            sample[key] = utils.sp_sparse_to_pt_sparse(sample[key])
        elif class_name.find('numpy') > 0:
            sample[key] = torch.from_numpy(sample[key])
        if args.model == "lap":
            sample["Di"] = None
            sample["DiA"] = None
            sample["flat_Di"] = None
            sample["flat_DiA"] = None

print("Preprocess Dataset")
for i in tqdm.tqdm(range(len(train_data))):
    convert(train_data[i])

for i in tqdm.tqdm(range(len(test_data))):
    convert(test_data[i])

def sample_batch(samples):
    indices = []
    for b in range(args.batch_size):
        ind = np.random.randint(0, len(samples))
        sample_batch.num_vertices = max(sample_batch.num_vertices, samples[ind]['V'].size(0))
        sample_batch.num_faces = max(sample_batch.num_faces, samples[ind]['F'].size(0))
        indices.append(ind)

    inputs = torch.zeros(args.batch_size, sample_batch.num_vertices, 3)
    mask = torch.zeros(args.batch_size, sample_batch.num_vertices, 1)

    flat_inputs = torch.zeros(args.batch_size, sample_batch.num_vertices, 3)
    faces = torch.zeros(args.batch_size, sample_batch.num_faces, 3).long()
    laplacian = []
    flat_laplacian = []

    Di = []
    DiA = []
    flat_Di = []
    flat_DiA = []

    for b, ind in enumerate(indices):
        num_vertices = samples[ind]['V'].size(0)
        num_faces = samples[ind]['F'].size(0)

        inputs[b, :num_vertices] = samples[ind]['V']
        flat_inputs[b, :num_vertices, 0:2] = samples[ind]['V'][:, 0:2]

        mask[b, :num_vertices] = 1
        faces[b, :num_faces] = samples[ind]['F']

        laplacian.append(samples[ind]['L'])
        flat_laplacian.append(samples[ind]['flat_L'])

        if args.model == "dir":
            Di.append(samples[ind]['Di'])
            DiA.append(samples[ind]['DiA'])

            flat_Di.append(samples[ind]['flat_Di'])
            flat_DiA.append(samples[ind]['flat_DiA'])



    if args.model == "dir":
        Di = utils.sparse_cat(Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices)
        DiA = utils.sparse_cat(DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces)

        flat_Di = utils.sparse_cat(flat_Di, 4 * sample_batch.num_faces, 4 * sample_batch.num_vertices)
        flat_DiA = utils.sparse_cat(flat_DiA, 4 * sample_batch.num_vertices, 4 * sample_batch.num_faces)
        return Variable(inputs).cuda(), Variable(flat_inputs).cuda(), Variable(mask).cuda(), Variable(laplacian).cuda(), Variable(flat_laplacian).cuda(), Variable(Di).cuda(), Variable(DiA).cuda(), Variable(flat_Di).cuda(), Variable(flat_DiA).cuda(), faces
    else:
        laplacian = utils.sparse_diag_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices)
        flat_laplacian = utils.sparse_diag_cat(flat_laplacian, sample_batch.num_vertices, sample_batch.num_vertices)
        return Variable(inputs).cuda(), Variable(flat_inputs).cuda(), Variable(mask).cuda(), Variable(laplacian).cuda(), Variable(flat_laplacian).cuda(), None, None, None, None, faces

sample_batch.num_vertices = 0
sample_batch.num_faces = 0

if args.model == "lap":
    model = LapVAE()
else:
    model = DirVAE()

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

fixed_noise_ = torch.FloatTensor(args.batch_size, 1, 100).normal_(0, 1)
if args.cuda:
    fixed_noise_ = fixed_noise_.cuda()
fixed_noise_ = Variable(fixed_noise_)

def log_normal_diag(z, mu, logvar):
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp())

def loss_function(recon_mu, recon_logvar, mask, x, z, mu, logvar):
    x = x.view(x.size(0), -1)
    recon_mu = recon_mu.view(x.size(0), -1)
    recon_logvar = recon_logvar.view(x.size(0), -1)

    mask = mask.repeat(1, 1, 3)
    mask = mask.view(x.size(0), -1)
    BCE = -(log_normal_diag(x, recon_mu, recon_logvar) * mask).sum(1).mean()

    log_q = log_normal_diag(z, mu, logvar)
    log_p = log_normal_diag(z, z * 0, z * 0)

    KLD_element = log_q - log_p
    KLD = KLD_element.sum(1).mean()
    return BCE, KLD

def main():
    for epoch in range(args.num_epoch):
        #torch.save(model, 'models/{}_conv.pt'.format(args.model))

        
        model.train()
        loss_value = 0.0
        loss_bce = 0.0
        loss_kld = 0.0

        # Train
        for j in tqdm.tqdm(range(len(train_data) // args.batch_size)):
            inputs, flat_inputs, mask, laplacian, flat_laplacian, Di, DiA, flat_Di, flat_DiA, faces = sample_batch(train_data)

            if args.model == "lap":
                recon_mu, recon_logvar, z, mu, logvar = model(inputs, flat_inputs, laplacian, flat_laplacian, mask)
            else:
                recon_mu, recon_logvar, z, mu, logvar = model(inputs, flat_inputs, Di, DiA, flat_Di, flat_DiA, mask)

            BCE, KLD = loss_function(recon_mu, recon_logvar, mask, inputs, z, mu, logvar)
            loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
            loss = BCE + KLD * min(epoch/10.0, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

        folder_name = 'mesh_mnist/results_' + args.model
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for k in range(10):
            mesh.save_as_ply(
                 folder_name + '/train_epoch_%03d_%03d_input.ply' % (k, epoch), inputs.data[k].cpu(), faces[k].cpu())
            mesh.save_as_ply(
                 folder_name + '/train_epoch_%03d_%03d_recon.ply' % (k, epoch), recon_mu.data[k].cpu(), faces[k].cpu())


        print("Train epoch {}, loss {}, bce {}, kld {}".format(epoch,
                loss_value / (len(train_data) // args.batch_size),
                loss_bce / (len(train_data) // args.batch_size),
                loss_kld / (len(train_data) // args.batch_size)))

        #model.eval()
        loss_value = 0.0
        loss_bce = 0.0
        loss_kld = 0.0
        

        # Evaluate
        for j in tqdm.tqdm(range(len(test_data) // args.batch_size)):
            inputs, flat_inputs, mask, laplacian, flat_laplacian, Di, DiA, flat_Di, flat_DiA, faces = sample_batch(test_data)

            if args.model == "lap":
                recon_mu, recon_logvar, z, mu, logvar = model(inputs, flat_inputs, laplacian, flat_laplacian, mask)
            else:
                recon_mu, recon_logvar, z, mu, logvar = model(inputs, flat_inputs, Di, DiA, flat_Di, flat_DiA, mask)

            BCE, KLD = loss_function(recon_mu, recon_logvar, mask, inputs, z, mu, logvar)
            loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
            loss = BCE + KLD

            loss_value += loss.item()

        for k in range(10):
            mesh.save_as_ply(
                 folder_name + '/test_epoch_%03d_%03d_input.ply' % (k, epoch), inputs.data[k].cpu(), faces[k].cpu())
            mesh.save_as_ply(
                 folder_name + '/test_epoch_%03d_%03d_recon.ply' % (k, epoch), recon_mu.data[k].cpu(), faces[k].cpu())


        _, fixed_flat_inputs, fixed_mask, _, fixed_flat_laplacian, _, _, fixed_flat_Di, fixed_flat_DiA, fixed_faces = sample_batch(test_data)
        fixed_noise = fixed_noise_.repeat(1, fixed_flat_inputs.size(1), 1)


        if args.model == "lap":
            fake, _ = model.decoder(fixed_flat_inputs, fixed_noise, fixed_flat_laplacian, fixed_mask)
        else:
            fake, _ = model.decoder(fixed_flat_inputs, fixed_noise, fixed_flat_Di, fixed_flat_DiA, fixed_mask)

        print("Test epoch {}, loss {}, bce {}, kld {}".format(epoch,
                        loss_value / (len(test_data) // args.batch_size),
                        loss_bce / (len(test_data) // args.batch_size),
                        loss_kld / (len(test_data) // args.batch_size)))

        for k in range(10):
            mesh.save_as_ply(
                 folder_name + '/samples_epoch_%03d_%03d.ply' % (k, epoch), fake.data[k].cpu(), fixed_faces[k].cpu())
            mesh.save_as_ply(
                 folder_name + '/real_epoch_%03d_%03d.ply' % (k, epoch), inputs.data[k].cpu(), faces[k].cpu())


if __name__ == "__main__":
    main()
