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
from multiprocessing.pool import Pool
from collections import namedtuple
import pickle
import time

mypath = "as_rigid_as_possible/data_obj/"
files = sorted([f for f in listdir(mypath) if not isfile(join(mypath, f))])

def process(seqname):
    print(seqname)
    id_, seqname = seqname
    sequence_name = mypath + seqname
    new_sequence = []

    for i in range(50):
        file_name = sequence_name + "/{0:02d}.obj".format(i)
        V, F = mesh.load_obj(file_name)

        if i < 10:
            dist = mesh.dist(V, F)
            areas = mesh.area(F, dist)
            W, A = mesh.cotangent_weights(F, areas, dist)
            D = sp.sparse.diags(W.sum(0).A.squeeze(), 0)

            L = graph.laplacian(W, symmetric=False, normalized=False)
            L = A * L

            #Di, DiA = mesh.fast_dirac(V, F)
            Di, DiA = mesh.dirac(V, F)

            new_sequence.append({'V': V.astype('float32'),
                                 'F':  F.astype('int32'),
                                 'L': L.astype('float32'),
                                 'Di': Di.astype('float32'),
                                 'DiA': DiA.astype('float32')})
        else:
            new_sequence.append({'V': V.astype('float32'),
                                 'F':  F.astype('int32')})

    np.save(open('as_rigid_as_possible/data_plus/{}.npy'.format(seqname), 'wb'), new_sequence)

if __name__ == "__main__":
    print("Loading the dataset")
    pool = Pool()
    pool.map(process, list(enumerate(files)))
    #process(files[0])
