'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

import glob
import os
import sys
sys.path.insert(0, os.path.expanduser('~/Workspace/libigl/python'))
scans = glob.glob(os.path.expanduser('~/Correspondance/MPI-FAUST/training/tet_wild/*.obj'))
from iglhelpers import p2e,e2p
import pyigl as igl
import scipy.sparse as sparse
import numpy as np

class Geom():
    def __init__(self, s=None):
        self.V = igl.eigen.MatrixXd()
        self.F = igl.eigen.MatrixXi()
        if s is not None:
            igl.read_triangle_mesh(s, self.V, self.F)

# Picking Largest Component
from scipy import stats
for path in scans:
    Inmodel = Geom(path)
    C = igl.eigen.MatrixXi()
    igl.facet_components(Inmodel.F, C)

    C = (e2p(C)).flatten()
    modeC, mode_count = stats.mode(C)

    if mode_count == Inmodel.F.rows():
        print(f"Already Fit  {path[-15:-12]}")
        igl.write_triangle_mesh(path[:-12]+'_single.obj', Inmodel.V, Inmodel.F)
    else:
        Fid = p2e(np.where(C==modeC)[0])
        F = igl.slice(Inmodel.F, Fid, 1)
        Outmodel = Geom()
        I, J = igl.eigen.MatrixXi(), igl.eigen.MatrixXi()
        igl.remove_unreferenced(Inmodel.V, F, Outmodel.V, Outmodel.F, I, J)
        igl.write_triangle_mesh(path[:-12]+'_single.obj', Outmodel.V, Outmodel.F)
        print(f'Written { path[-15:-12]}')