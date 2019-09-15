import sys,os
import glob

sys.path.append(os.path.expanduser('~/Workspace/libigl/python/'))
import pyigl as igl
from iglhelpers import p2e,e2p

import numpy as np
import scipy

Xd = igl.eigen.MatrixXd
Xi = igl.eigen.MatrixXi

def vertex_area(V,F):
    if type(V).__module__ == np.__name__:
        V,F = p2e(V), p2e(F)
    M = igl.eigen.SparseMatrixd()
    igl.massmatrix((V), (F), igl.MASSMATRIX_TYPE_VORONOI, M)
    return e2p(M.diagonal())

def face_area(V,F):
    if type(V).__module__ == np.__name__:
        V,F = p2e(V), p2e(F)
    A = Xd()
    igl.doublearea((V), (F),A)
    A = e2p(A).flatten()
    return A

def fix_degen(eV,eF):
    # duplicate vertex removal
    if np.min(face_area(eV,eF)) > 1e-15 and np.min(vertex_area(eV,eF)) > 1e-15:
        return e2p(eV), e2p(eF), None, None
    
    nV, nF = Xd(), Xi()
    SVI, SVJ = Xi(), Xi()
    V, F = e2p(eV), e2p(eF)
    SVI,  SVJ= None,None
    short_f, short_e = np.where(np.asarray([edge_lens(V[f]) for f in F]) < 1e-15)
    if len(short_f) > 0:
        V,F = collapse_edge(V,F,short_f, short_e)
    V,F = fix_degen_face_with_flip(eV, eF)
    if V is None:
        assert False
    return V,F,SVI,SVJ
    
def fix_degen_face_with_flip(V,F):
    # Flip degenerate face.
    eFF,eFFi = Xi(), Xi()
    eA = igl.eigen.SparseMatrixi()
    if type(V).__module__ == np.__name__:
        eV,eF = p2e(V), p2e(F)
    else:
        eV, eF = V,F
        V,F = e2p(eV), e2p(eF)
    igl.triangle_triangle_adjacency(eF, eFF,eFFi)
    igl.adjacency_matrix(eF, eA)

    FF, FFi = e2p(eFF), e2p(eFFi)
    Adj = e2p(eA).tolil()
    
    for _ in range(F.shape[0]):
        degen_f = np.where(face_area(V,F) < 1e-15)[0]
        if len(degen_f) == 0:
            break
        fid = degen_f[0]
        lens = edge_lens(V[F[fid]])
        if np.min(lens) < 1e-15:
            assert False
        e0 = np.argmax(lens)
        if not edge_flip(F, FF, FFi, fid, e0, Adj):
            print(degen_f)
    else:
        assert False
    return V,F

edge_lens = lambda verts: np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1) # edge length

# numpy function
def edge_flip(F, FF, FFi, f0, e0, AdjMat_lil):
    f1 = int(FF[f0, e0])
    if f1 == -1:
        assert False
    e1 = int(FFi[f0, e0])
    e01 = (e0 + 1) % 3
    e02 = (e0 + 2) % 3
    e11 = (e1 + 1) % 3
    e12 = (e1 + 2) % 3
    f01 = int(FF[f0, e01])
    f02 = int(FF[f0, e02])
    f11 = int(FF[f1, e11])
    f12 = int(FF[f1, e12])

    u1 = F[f0, e01]
    u0 = F[f1, e11]
    v0 = F[f0, e02]
    v1 = F[f1, e12]

    # topology constraint: check if v0 v1 already exists in F
    if AdjMat_lil[v0, v1] != 0:
        assert False

    AdjMat_lil[v0, v1] = 1
    AdjMat_lil[v1, v0] = 1
    AdjMat_lil[u0, u1] = 0
    AdjMat_lil[u1, u0] = 0

    F[f0, e01] = F[f1, e12]
    F[f1, e11] = F[f0, e02]
    FF[f0, e0] = f11
    FF[f0, e01] = f1
    FF[f1, e1] = f01
    FF[f1, e11] = f0
    if f11 != -1:
        FF[f11, FFi[f1, e11]] = f0
    if f01 != -1:
        FF[f01, FFi[f0, e01]] = f1

    FFi[f0, e0] = FFi[f1, e11]
    FFi[f1, e1] = FFi[f0, e01]
    FFi[f0, e01] = e11
    FFi[f1, e11] = e01

    if f11 != -1:
        FFi[f11, FFi[f0, e0]] = e0
    if f01 != -1:
        FFi[f01, FFi[f1, e1]] = e1
    return True