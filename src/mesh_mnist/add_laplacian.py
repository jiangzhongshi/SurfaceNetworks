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
from multiprocessing.pool import Pool
from collections import namedtuple
import pickle
import time


def process(sample):
    ind, sample = sample
    print(ind)
    V, F, label = sample['V'], sample['F'], sample['label']
    V = V / 27
    V -= np.array([0.5, 0.5, 0.0])

    dist = mesh.dist(V, F)

    areas = mesh.area(F, dist)
    W, A = mesh.cotangent_weights(F, areas, dist)
    L = graph.laplacian(W, symmetric=False, normalized=False) # Return to normalized=True
    L = A * L # Comment this and uncomment above

    Di, DiA = mesh.dirac(V, F)

    V_ = V.copy()
    V_[:, 2] = 0
    dist = mesh.dist(V_, F)

    areas = mesh.area(F, dist)
    W, A = mesh.cotangent_weights(F, areas, dist)
    flat_L = graph.laplacian(W, symmetric=False, normalized=False) # Return to normalized=True
    flat_L = A * flat_L 

    flat_Di, flat_DiA = mesh.dirac(V_, F)

    return {'V': V.astype('float32'),
            'F':  F.astype('int32'),
            'L': L.astype('float32'),
            'flat_L': flat_L.astype('float32'),
            'Di': Di.astype('float32'),
            'DiA': DiA.astype('float32'),
            'flat_Di': flat_Di.astype('float32'),
            'flat_DiA': flat_DiA.astype('float32'),
            'label': label}

if __name__ == "__main__":
    print("Loading the dataset")
    train_data = np.load(open("mesh_mnist/data/train.np", "rb"))
    test_data = np.load(open("mesh_mnist/data/test.np", "rb"))

    pool = Pool()

    train_data_plus = pool.map(process, enumerate(train_data))
    test_data_plus = pool.map(process, enumerate(test_data))

    np.save(open('mesh_mnist/data/train_plus.np', 'wb'), train_data_plus)
    np.save(open('mesh_mnist/data/test_plus.np', 'wb'), test_data_plus)
