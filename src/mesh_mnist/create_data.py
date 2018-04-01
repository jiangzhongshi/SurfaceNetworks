'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

from matplotlib import pyplot as plt
import seaborn
from math import sqrt, pi
from poisson_disc import Grid
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from scipy.spatial import Delaunay
from utils import mesh
import gc
from multiprocessing.pool import Pool

def bilinear_interpolation(image, y, x):
    f00 = image[27-int(x), int(y)]
    f01 = image[27-int(x), int(y+1)]
    f10 = image[27-int(x+1), int(y)]
    f11 = image[27-int(x+1), int(y+1)]

    dx = x-int(x)
    dy = y-int(y)
    return f00 * (1-dx) * (1-dy) + f01 * (1-dx) * dy + f10 * dx * (1-dy) + f11 * dx * dy

train = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

test = datasets.MNIST('../data', train=False, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ]))

print(train.train_data.size())
print(test.test_data.size())

if False:
    images = test.test_data
    labels = test.test_labels
    name = 'test'
else:
    images = train.train_data
    labels = train.train_labels
    name = 'train'

def run(j):
    print(j)
    while True:
        r = 1.5
        length = 27
        width = 27
        grid = Grid(r, length, width)

        rand = (random.uniform(0, length), random.uniform(0, width))
        data = np.array(grid.poisson(rand))

        if len(data) > 100:
            def unzip(items):
                return ([item[i] for item in items] for i in range(len(items[0])))

            colors = []
            for d in data:
                colors.append(bilinear_interpolation(images[j], d[0], d[1]))

            tri = Delaunay(data)
            data = tri.points

            colors = np.expand_dims(np.array(colors), 1)
            V = np.concatenate([data, colors/255], 1).astype('float32')
            F = np.array(tri.simplices).astype('int32')
            dists = mesh.dist(V, F)

            areas = mesh.area(F, dists)

            V_ = V.copy()
            V_[:, 2] = 0
            dists = mesh.dist(V_, F)
            areas_ = mesh.area(F, dists)

            if np.min(areas) > 1e-2 and np.min(areas_) > 1e-2:
                break

    return {'V': V, 'F': F, 'label': labels[j]}


pool = Pool()
dataset = pool.map(run, list(range(len(images))))

np.save(open('mesh_mnist/data/{}.np'.format(name), 'wb'), dataset)
