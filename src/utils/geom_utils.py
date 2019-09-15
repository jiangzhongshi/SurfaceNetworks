import sys
import os
import numpy as np
sys.path.append(os.path.expanduser('~/Workspace/libigl/python'))
import pyigl as igl
from iglhelpers import e2p, p2e
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm

def invert_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

def laplacian_eigendecomp(V,F, k=30):
    eV, eF = p2e(V), p2e(F)

    L = igl.eigen.SparseMatrixd()
    igl.cotmatrix(eV,eF,L)

    M, Minv = igl.eigen.SparseMatrixd(),igl.eigen.SparseMatrixd()
    igl.massmatrix(eV,eF,igl.MASSMATRIX_TYPE_VORONOI,M)
    igl.invert_diag(M, Minv)
    M = e2p(M)
    Minv = e2p(Minv)
    Lc = e2p(L)
    lap_eigs = scipy.sparse.linalg.eigsh(Lc.tocsc(),
    M=M.tocsc(),
    k=k, sigma=0, which='LM')
    return lap_eigs


def triangle_normal_area(v0,v1,v2):
    v0, v1, v2 = np.asarray(v0),np.asarray(v1),np.asarray(v2)
    tna = np.cross(v1-v0, v2-v0)/2
    return tna

def edge_flip_with_deg_geom(V, F, FF, FFi, f0, e0, AdjMat, V_deg = None):
    f1 = int(FF[f0, e0])
    if f1 == -1: return False, 'bnd'
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

    # Mandatory: Topology checker. (v0,v1) cannot already exist
    if AdjMat[v0, v1] != 0: return False, 'AM'

    # degree checker
    for u in [u0,u1]:
        if V_deg[u] < 5: return False, 'd5'
    for v in [v0, v1]:
        if V_deg[v] > 7: return False, 'd7'

    # Geometry Checker: (N_f0, N_f1)/2 w/ ((v0,v1,u0), (v0, u1, v1))/2
    N_f0 = triangle_normal_area(V[u1], V[u0], V[v0])
    N_f1 = triangle_normal_area(V[u1], V[v1], V[u0])
    N_f1_ = triangle_normal_area(V[u1], V[v1], V[v0])
    N_f0_ = triangle_normal_area(V[v1], V[u0], V[v0])
    normalize = lambda x: x/np.linalg.norm(x)
    if np.linalg.norm(N_f0_) < 1e-10 or np.linalg.norm(N_f1_) < 1e-10:
        return False, 'area'
    angle = np.dot(normalize(N_f0+ N_f1), normalize(N_f0_ + N_f1_))
    if angle < 0.5: # more than 60 degree
        return False, 'angle'

    # Perform Flip
    for u in [u0,u1]:
        V_deg[u] = V_deg[u] - 1
    for v in [v0, v1]:
        V_deg[v] = V_deg[u] + 1
    AdjMat[v0, v1] = 1
    AdjMat[v1, v0] = 1
    AdjMat[u0, u1] = 0
    AdjMat[u1, u0] = 0

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
    return True, 'Success'

# numpy function
def edge_flip(F, FF, FFi, f0, e0, AdjMat_lil):
    f1 = int(FF[f0, e0])
    if f1 == -1:
        return False
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
        return False

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

def quaternion_matrix(x):
    a, b, c, d = x.tolist()
    return np.array([[a, -b, -c, -d],
                     [b,  a, -d,  c],
                     [c,  d,  a, -b],
                     [d, -c,  b,  a]])

def normalized_laplacian(V, F): # input ev, eF
    V = p2e(V.astype(np.float64))
    F = p2e(F.astype(np.int32))
    L_igl = igl.eigen.SparseMatrixd()
    M, Minv =  igl.eigen.SparseMatrixd(),igl.eigen.SparseMatrixd()
    igl.cotmatrix(V,F,L_igl)
    igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_BARYCENTRIC,M)
    pM = e2p(M)
    if np.any(pM.diagonal() == 0):
        return None
    igl.invert_diag(M,Minv)
    L_igl = Minv*L_igl
    L = e2p(L_igl).astype(np.float32)
    L = L/spnorm(L)
    return L.astype(np.float32)

# igl function
def gaussian_curvature(V,F, area_avg=False):
    V = igl.eigen.MatrixXd(V)
    F = p2e(F)
    K = V.copy()
    igl.gaussian_curvature(V,F,K)
    K = e2p(K)
    if area_avg:
        M = igl.eigen.SparseMatrixd()
        igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_VORONOI,M)
        M = e2p(M).diagonal()
        K *= (1/M)
    return (K)



def hacky_compute_laplacian(V, F, hack=1):
    V, F = p2e(V), p2e(F)
    L_igl = igl.eigen.SparseMatrixd()
    M, Minv =  igl.eigen.SparseMatrixd(),igl.eigen.SparseMatrixd()
    igl.cotmatrix(V,F,L_igl)
    igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_BARYCENTRIC,M)
    igl.invert_diag(M,Minv)
    L_igl = Minv*L_igl
    L = e2p(L_igl).astype(np.float32)
    L.data[np.where(np.logical_not(np.isfinite(L.data)))[0]] = hack
    L.data[L.data > 1e10] = hack
    L.data[L.data < -1e10] = hack
    return L.tocsr().tocoo()

# igl function
def compute_laplacian(V, F, bary=False):
    V = igl.eigen.MatrixXd(V)
    F = igl.eigen.MatrixXi(F)
    L_igl = igl.eigen.SparseMatrixd()
    M, Minv =  igl.eigen.SparseMatrixd(),igl.eigen.SparseMatrixd()
    igl.cotmatrix(V,F,L_igl)
    if bary:
        masstype = igl.MASSMATRIX_TYPE_BARYCENTRIC
    igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_VORONOI,M)
    pM = e2p(M)
    if np.any(pM.diagonal() == 0):
        return None
    igl.invert_diag(M,Minv)
    L_igl = Minv*L_igl
    return e2p(L_igl).astype(np.float32)

# igl function
def laplacian_and_mass(V, F, bary=False):
    V, F = p2e(V), p2e(F)
    L_igl = igl.eigen.SparseMatrixd()
    M, Minv =  igl.eigen.SparseMatrixd(),igl.eigen.SparseMatrixd()
    igl.cotmatrix(V,F,L_igl)
    masstype = igl.MASSMATRIX_TYPE_VORONOI
    igl.massmatrix(V,F,masstype,M)
    pM = e2p(M).diagonal()
    if np.any(np.isnan(pM)) or np.any(pM <= 0):
        return (None, None)
    igl.invert_diag(M,Minv)
    L_igl = Minv*L_igl
    return e2p(L_igl).astype(np.float32), pM

def area(V, F):
    V = p2e(V)
    F = p2e(F)
    dblAf = igl.eigen.MatrixXd()
    igl.doublearea(V,F,dblAf)
    return e2p(dblAf).flatten()/2

# np-igl
def dirac(V, F):
    V = igl.eigen.MatrixXd(V)
    F = igl.eigen.MatrixXi(F)
    dblAf = igl.eigen.MatrixXd()
    Di, DiA = igl.eigen.SparseMatrixd(), igl.eigen.SparseMatrixd()
    igl.dirac_operator(V,F,Di,DiA)
    return e2p(Di), e2p(DiA)


#def dirac(V, F):
#    V = igl.eigen.MatrixXd(V)
#    F = igl.eigen.MatrixXi(F)
#    dblAf = igl.eigen.MatrixXd()
#    igl.doublearea(V,F,dblAf)
#
#    D = np.zeros((4 * F.rows(), 4 * V.rows()), dtype = np.float32)
#    DA = np.zeros((4 * V.rows(), 4 * F.rows()), dtype = np.float32)
#
#    dblAv = np.zeros(V.rows())
#    for i in range(F.rows()):
#        for j in F.row(i):
#            dblAv[j] += dblAf[i] / 3
#
#    for i in range(F.rows()):
#        for ind, j in enumerate(F.row(i)):
#            ind1 = F[i, (ind + 1) % 3]
#            ind2 = F[i, (ind + 2) % 3]
#
#            e1 = V.row(ind1)
#            e2 = V.row(ind2)
#
#            e = np.array([0, e1[0] - e2[0], e1[1] - e2[1], e1[2] - e2[2]], dtype = np.float32)
#
#            mat = -quaternion_matrix(e) / (dblAf[i])
#            D[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = mat
#            DA[j * 4:(j + 1) * 4, i * 4: (i + 1) * 4] = mat.transpose() * dblAf[i] / dblAv[j]
#
#    D = scipy.sparse.csr_matrix(D)
#    DA = scipy.sparse.csr_matrix(DA)
#
#    return D.astype(np.float32), DA.astype(np.float32)


# np
def unit_bounding_box(A): # Dirac scale down, vdist scale up with V+
    upper = A.max(axis = 0)
    lower = A.min(axis = 0)
    center = (upper+lower)/2
    scale = (upper-lower).max()
    A = (A - center) / np.float32(scale) # numpy syntax sugar
    return A, scale # scale of input

def permute_CSR_matrix(M, order):
    permuted_row = order[M.row]
    permuted_col = order[M.col]
    new_M = scipy.sparse.csr_matrix((M.data, (permuted_row, permuted_col)), shape=M.shape)
    return new_M

def dual_adjacency(F,TT=None):
    eF = p2e(F)
    if TT is None:
        TT, TTi = igl.eigen.MatrixXi(), igl.eigen.MatrixXi()
        igl.triangle_triangle_adjacency(eF,TT,TTi)
        TT = e2p(TT)
    A = scipy.sparse.dok_matrix((F.shape[0], F.shape[0]))
    for i, tt in enumerate(TT):
        #A[i, i] = 1
        tt = [t for t in tt if t != -1]
        for t in tt:
            A[i, t] = 1
    return A.tocsr()

def adjacency(F):
    A = igl.eigen.SparseMatrixi()
    igl.adjacency_matrix(p2e(F), A)
    return e2p(A)

def left_normalized_adjacency(V,F):
    A = adjacency(F)
    vdeg = np.asarray(A.sum(axis=0)).flatten()
    deg_invsq = np.power(vdeg,-1)
    deg_invsq[deg_invsq == np.inf] = 0
    D_invsq = sp.diags(deg_invsq, dtype=np.float32)

    return D_invsq*A

def normalized_adjacency(V,F):
    A = adjacency(F)
    vdeg = np.asarray(A.sum(axis=0)).flatten()
    deg_invsq = np.power(vdeg,-0.5)
    deg_invsq[deg_invsq == np.inf] = 0
    D_invsq = sp.diags(deg_invsq, dtype=np.float32)

    return D_invsq*A*D_invsq

def sym_norm_Laplacian(V,F):
    v_num = V.shape[0]

    SNLap = sp.identity(v_num) - normalized_adjacency(V,F)
    return SNLap

def clipOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    a[a < quartileSet[0]] = quartileSet[0]
    a[a > quartileSet[1]] = quartileSet[1]
    return a

def constrained_edge_flip(V, F, num_flipped_edges):
    TT, TTi = igl.eigen.MatrixXi(), igl.eigen.MatrixXi()
    igl.triangle_triangle_adjacency(p2e(F),TT,TTi)

    AdjMat = adjacency(F)
    AdjMat = AdjMat.tolil().astype(np.float32)
    V_deg = AdjMat.getnnz(axis=0)
    fail_list = []
    for f,e in zip(np.random.randint(0, F.shape[0],  size=(num_flipped_edges,)),
            np.random.randint(0, 3, size=(num_flipped_edges,))):
            flag, reason = edge_flip_with_deg_geom(V, F, TT, TTi, f, e, AdjMat, V_deg)
            if not flag:
                fail_list.append(reason)
                continue # maintain manifold and constraint valence
    #print(f'{len(fail_list)}:{fail_list}')
    return V, F

def compute_normals(pV,pF):
    V = igl.eigen.MatrixXd(pV)
    F = igl.eigen.MatrixXi(pF.astype(np.int32))
    N = igl.eigen.MatrixXd()

    igl.per_vertex_normals(V,F,N)
    return e2p(N)

def compute_curv4(pV,pF):
    Xd = igl.eigen.MatrixXd
    V = igl.eigen.MatrixXd(pV)
    F = igl.eigen.MatrixXi(pF.astype(np.int32))
    N = igl.eigen.MatrixXd()
    PD1,PD2, PV1, PV2 = Xd(), Xd(), Xd(), Xd()
    igl.principal_curvature(V,F,PD1,PD2,PV1,PV2)
    PV1, PV2 = e2p(PV1), e2p(PV2)
    H = 0.5*(PV1+PV2)
    G = PV1 * PV2
    stack= np.hstack([PV1, PV2, H,G])
    stack = np.clip(stack, -100, 100)
    if np.isnan(stack).any():
        return None
    stack /= np.max(np.abs(stack), axis=0)
    return stack

def compute_wks(pV,pF, N=100):
    V = igl.eigen.MatrixXd(pV)
    F = igl.eigen.MatrixXi(pF.astype(np.int32))
    L = igl.eigen.SparseMatrixd()
    M = igl.eigen.SparseMatrixd()
    igl.cotmatrix(V,F,L)
    igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_VORONOI,M)

    Am = np.clip(e2p(M).tocoo().data, 1e-8,np.inf)
    Am /= np.sum(Am)
    Aminv = scipy.sparse.diags(1/Am)
    num_v = V.rows()

    E, phi = scipy.sparse.linalg.eigsh(-e2p(L), M = scipy.sparse.diags(Am), sigma=-1e-5, k=300)
    #E, phi = scipy.sparse.linalg.eigsh(-Aminv*e2p(L), sigma=-1e-5, k=300)

    E = np.abs(np.real(E))
    phi = np.real(phi)

    E_ind = np.argsort(E)
    E = E[E_ind]
    phi = phi[:,E_ind]
    logE = np.log(np.clip(E, 1e-6, np.inf)).T

    ee = np.linspace(logE[1],max(logE)/1.02, N)
    sigma = (ee[1] - ee[0])*6

    C = np.zeros(N)
    WKS = np.zeros((num_v, N))
    for i in range(N):
        exp_sth = np.exp((-(ee[i]-logE)**2)/(2*sigma**2))
        C[i] = np.sum(exp_sth)
        WKS[:,i] = np.dot(phi**2, exp_sth)
    return np.divide(WKS, np.tile(C.T,(num_v,1)))

def pca_whiten(V):
    V -= np.mean(V,axis=0)
    _,_,PCA = np.linalg.svd(V)
    V = V @np.linalg.inv(PCA)
    V = rescale_V(V)
    return V


def rescale_V(V):
    V -= np.min(V,axis=0)
    V /= np.max(V)
    return V
