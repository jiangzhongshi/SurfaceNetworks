'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import scipy as sp
import itertools
from scipy import sparse
import utils.graph as graph

def dist(V, F):
    num_vertices = V.shape[0]
    W = np.zeros((num_vertices, num_vertices))

    for face in F:
        vertices = face.tolist()
        for i, j in itertools.product(vertices, vertices):
            W[i, j] = np.sqrt(((V[i] - V[j]) ** 2).sum())

    return sparse.csr_matrix(W)

def quaternion_matrix(x):
    a, b, c, d = x.tolist()
    return np.array([[a, -b, -c, -d],
                     [b,  a, -d,  c],
                     [c,  d,  a, -b],
                     [d, -c,  b,  a]])

def dirac(V, F):
    l = dist(V, F)
    Af = area(F, l)
    Av = np.zeros(V.shape[0])

    D = np.zeros((4 * F.shape[0], 4 * V.shape[0]))
    DA = np.zeros((4 * V.shape[0], 4 * F.shape[0]))

    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            Av[j] += Af[i] / 3

    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            ind1 = F[i, (ind + 1) % 3]
            ind2 = F[i, (ind + 2) % 3]

            e1 = V[ind1]
            e2 = V[ind2]

            e = np.array([0, e1[0] - e2[0], e1[1] - e2[1], e1[2] - e2[2]])

            mat = -quaternion_matrix(e) / (2 * Af[i])
            D[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = mat
            DA[j * 4:(j + 1) * 4, i * 4: (i + 1) * 4] = mat.transpose() * Af[i] / Av[j]

    D = sparse.csr_matrix(D)
    DA = sparse.csr_matrix(DA)

    return D, DA


def area(F, l):
    areas = np.zeros(F.shape[0])

    for f in range(F.shape[0]):
        i, j, k = F[f].tolist()
        sijk = (l[i, j] + l[j, k] + l[k, i]) / 2

        sum_ = sijk * (sijk - l[i, j]) * (sijk - l[j, k]) * (sijk - l[k, i])
        if sum_ > 0:
            areas[f] = np.sqrt(sum_)
        else:
            areas[f] = 1e-6

    return areas

def uniform_weights(dist):
    W = sp.sparse.csr_matrix((1 / dist.data, dist.indices, dist.indptr), shape=dist.shape)

    # No self-connections.
    W.setdiag(0)
    W.eliminate_zeros()

    assert np.abs(W - W.T).mean() < 1e-10
    return W

def exp_weights(dist, sigma2):
    W = sp.sparse.csr_matrix((np.exp(-dist.data**2 / sigma2), dist.indices, dist.indptr), shape=dist.shape)

    # No self-connections.
    W.setdiag(0)
    W.eliminate_zeros()

    assert np.abs(W - W.T).mean() < 1e-10
    return W

def cotangent_weights(F, a, l):
    W = np.zeros(l.shape)
    A = np.zeros(l.shape[0])

    for f in range(F.shape[0]):
        for v_ind in itertools.permutations(F[f].tolist()):
            i, j, k = v_ind
            W[i, j] += (-l[i, j]**2 + l[j, k]**2 + l[k, i]**2) / (8 * a[f] + 1e-6)
            A[i] += a[f] / 3 / 4 # each face will appear 4 times

    return sp.sparse.csr_matrix(W), sp.sparse.diags(1/(A+1e-9), 0)

def laplacian(W, A_inv):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    D = sp.sparse.diags(d.A.squeeze(), 0)
    L = A_inv * (D - W)

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sp.sparse.csr.csr_matrix
    return L


def ply_to_numpy(plydata):
    V = np.stack([plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']])
    V = V.transpose(1, 0)

    F = plydata['face'].data['vertex_indices']

    F = np.stack(F).astype('int32')
    return V, F

def save_as_ply(filename, V, F):
    output = \
"""ply
format ascii 1.0
comment Created by Blender 2.78 (sub 0) - www.blender.org, source file: ''
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar uint vertex_indices
end_header
""".format(V.size(0), F.size(0))

    for i in range(V.size(0)):
        output += "{} {} {}\n".format(V[i][0], V[i][1], V[i][2])

    for i in range(F.size(0)):
        output += "3 {} {} {}\n".format(F[i][0], F[i][1], F[i][2])

    text_file = open(filename, "w")
    text_file.write(output)
    text_file.close()


def save_as_obj(filename, V, F):
        output = ""

        for i in range(V.size(0)):
            if V[i].abs().sum() > 0:
                output += "v {} {} {}\n".format(V[i][0], V[i][1], V[i][2])

        for i in range(F.size(0)):
            if F[i].abs().sum() > 0:
                output += "f {} {} {}\n".format(F[i][0] + 1, F[i][1] + 1, F[i][2] + 1)

        text_file = open(filename, "w")
        text_file.write(output)
        text_file.close()

def adjacency_matrix_from_faces(F, num_vertices):
    A_v = np.zeros((num_vertices, num_vertices))
    A_f0 = np.zeros((num_vertices, num_vertices))
    A_f1 = np.zeros((num_vertices, num_vertices))

    for f in range(F.shape[0]):
        i, j, k = F[f].tolist()
        A_v[i, j] = A_v[j, i] = 1
        A_v[j, k] = A_v[k, j] = 1
        A_v[k, i] = A_v[i, k] = 1

        A_f[i][f] = 1
        A_f[j][f] = 1
        A_f[k][f] = 1

    return sp.sparse.csr_matrix(A_v), sp.sparse.csr_matrix(A_f)

def load_obj(filename):
    V = []  # vertex
    F = []  # face indexies

    fh = open(filename)
    for line in fh:
        if line[0] == '#':
            continue

        line = line.strip().split(' ')
        if line[0] == 'v':  # vertex
            V.append([float(line[i+1]) for i in range(3)])
        elif line[0] == 'f':  # face
            face = line[1:]
            for i in range(0, len(face)):
                face[i] = int(face[i].split('/')[0]) - 1
            F.append(face)

    V = np.array(V)
    F = np.array(F)

    return V, F

def centroids(V, F):
    C = np.zeros(F.shape)
    for i in range(F.shape[0]):
        C[i] = (V[F[i, 0]] + V[F[i, 1]] + V[F[i, 2]]) / 3
    return C

if __name__ == "__main__":
    from mayavi import mlab
    
    V, F = load_obj("bunny.obj")

    mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], F)

    dists = dist(V, F)

    areas = area(F, dists)

    W, _ = cotangent_weights(F, areas, dists)
    L = graph.laplacian(W, symmetric=False)
    C = centroids(V, F)

    D, DA = dirac(V, F)
    D = D.todense()
    DA = DA.todense()


    L = np.asarray(L.todense())
    P = np.asarray(np.matmul(L, V))
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    u, v, w = P[:, 0], P[:, 1], P[:, 2]

    V_ = np.pad(P, [[0, 0], [1, 0]], 'constant')
    V_ = V_.reshape(-1, 1)

    F_ = np.matmul(D, V_)

    F__ = np.matmul(DA, F_)

    F_ = np.array(F_.reshape(-1).reshape(F.shape[0], -1))
    F__ = np.array(F__.reshape(-1).reshape(V.shape[0], -1))

    mlab.quiver3d(C[:, 0], C[:, 1], C[:, 2], F_[:, 1], F_[:, 2], F_[:, 3], color=(0, 0, 1))
    mlab.quiver3d(V[:, 0], V[:, 1], V[:, 2], F__[:, 1], F__[:, 2], F__[:, 3], color=(0, 1, 0))

    mlab.show()
