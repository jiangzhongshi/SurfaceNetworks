'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Modified from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
'''

import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np

def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def laplacian(W, normalized=True, symmetric=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        if symmetric:
            d += np.spacing(np.array(0, W.dtype))
            d = 1 / np.sqrt(d)
            D = scipy.sparse.diags(d.A.squeeze(), 0)
            I = scipy.sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W * D
        else:
            d += np.spacing(np.array(0, W.dtype))
            d = 1.0 / d
            D = scipy.sparse.diags(d.A.squeeze(), 0)
            I = scipy.sparse.identity(d.size, dtype=W.dtype)
            L = I - D * W

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L
