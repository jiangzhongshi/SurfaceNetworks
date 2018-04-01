'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

import sys, os
import argparse

sys.path.insert(0, os.path.expanduser('~/Workspace/libigl/python'))
import pyigl as igl

from shared import TUTORIAL_SHARED_PATH, check_dependencies

dependencies = ["viewer"]
check_dependencies(dependencies)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--draw_gt',
                    action='store_true',
                    default=False,
                    help='sum the integers (default: find the max)')
args = parser.parse_args()
print(args)

V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()

V_other = igl.eigen.MatrixXd()
F_other = igl.eigen.MatrixXi()

V_full = igl.eigen.MatrixXd()
F_full = igl.eigen.MatrixXi()

seq_id = 0
time_step = 0

def pre_draw(viewer):
    global seq_id
    global time_step
 
    filename = "/Users/kostrikov/tmp/arap/results_lap/samples_epoch_{0:03d}_{1:03d}".format(seq_id, time_step % 20)
    if time_step < 20:
        filename += "_0curr.obj"
    else:
        filename += "_1pred.obj"        

    global V, F
    global V_other, F_other
    igl.readOBJ(filename, V, F)

    if filename.find('1pred') and args.draw_gt == True:
        filename_other = filename.replace('1pred', '2targ')
        igl.readOBJ(filename_other, V_other, F_other)
        
        V_full = igl.cat(1, V, V_other)
        F_full = igl.cat(1, F, F_other + V.rows())
    else:
        V_full = V
        F_full = F

    viewer.data.clear()
    viewer.data.set_mesh(V_full, F_full)
     
    C = igl.eigen.MatrixXd(F_full.rows(), 3)

    red = igl.eigen.MatrixXd([[1.0, 0.0, 0.0]])
    blue = igl.eigen.MatrixXd([[0.0, 0.0, 1.0]])
    green = igl.eigen.MatrixXd([[0.0, 1.0, 0.0]])

    for f in range(F_full.rows()):
        if time_step < 20:
            C.setRow(f, red)    
        else:
            if f < F.rows():
                C.setRow(f, blue)    
            else:
                C.setRow(f, green)    

    viewer.data.set_colors(C)

    time_step += 1
    if time_step == 40:
        time_step = 0
        seq_id += 1

        if seq_id == 64:
            seq_id = 0

    return False

viewer = igl.viewer.Viewer()
viewer.core.show_lines = False
viewer.core.is_animating = True
viewer.core.animation_max_fps = 20.0

viewer.callback_pre_draw = pre_draw
viewer.launch()