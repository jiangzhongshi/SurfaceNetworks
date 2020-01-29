import torch
import os
import sys
import igl
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import geom_utils
from utils import utils_pt as util
from models import DirDeepModel, LapDeepModel, IdDeepModel, AvgModel, MlpModel, LapMATModel, GatDeepModel, EfficientCascade
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import torch.nn.functional as F
import random
import re
import glob
import utils.mesh as mesh
from iglhelpers import e2p, p2e
from sklearn.externals import joblib

def read_npz(seq_names, args):
    with open(seq_names) as fp:
        V, _, VN, F,_,_ = igl.read_obj(seq_names)
        new_frame = {}
        npfloat = np.float32
        # Fix Degen,
        vdist = VN

        L, mass, Di, DiA, weight = None, None, None, None, None

        if not np.isfinite(vdist).all():
            print(f'warning: {seq_names} nan vdist')
            return None

        if 'hack1' in args.additional_opt:
            hack = 1
        elif 'hack0' in args.additional_opt:
            hack=0
        
        if 'intrinsic' in args.additional_opt:
            hack = None
        def hackit(Op, h):
            Op.data[np.where(np.logical_not(np.isfinite(Op.data)))[0]] = h
            Op.data[Op.data > 1e10] = h
            Op.data[Op.data < -1e10] = h
            return Op

        if args.uniform_mesh:
            V -= np.min(V, axis=0)
            V /= np.max(V) # isotropic scaling
        if args.model.startswith('dirac'):
            Di, DiA = geom_utils.dirac(V, F)
            Di = Di.astype(np.float32)
            DiA = DiA.astype(np.float32)
            Di = hackit(Di, hack)
            DiA = hackit(DiA, hack)
            Di, DiA = util.sp_sparse_to_pt_sparse(Di), util.sp_sparse_to_pt_sparse(DiA)
            new_frame['Di'] = Di
            new_frame['DiA'] = DiA
            if not (torch.isfinite(Di._values()).all() and torch.isfinite(DiA._values()).all()):
            # if np.isfinite(Di.data).all() and np.isfinite(DiA.data).all():
                print(f'warning: {seq_names} nan D')
                return None
        else:
            if L is None:
                if hack is None:
                    import ipdb;ipdb.set_trace()
                    L = mesh.intrinsic_laplacian(V,F)
                else:
                    L = geom_utils.hacky_compute_laplacian(V,F, hack)

            if L is None:
                print("warning: {} no L".format(seq_names))
                return None
            if np.any(np.isnan(L.data)):
                print(f"warning: {seq_names} nan L")
                return None
            new_frame['L'] = util.sp_sparse_to_pt_sparse(L.astype(np.float32))

        input_tensors = {}
        if 'V' in args.input_type:
            input_tensors['V'] = V

        new_frame['input'] = torch.cat([torch.from_numpy(input_tensors[t]) for t in input_tensors ], dim=1)

        # save data to new frame
        new_frame['V'] = V
        new_frame['F'] = F
        new_frame['target_dist'] = torch.from_numpy(vdist).view(-1,3)
        new_frame['name'] = seq_names
        return new_frame

def sample_batch(seq_names, args, is_fixed=False):
    sample_batch.num_vertices = 0
    sample_batch.num_faces = 0
    sample_batch.input_features = args.input_dim

    samples = []
    sample_names = []

    while len(samples) < args.batch_size:
        new_sample = None
        while True:
            if is_fixed:
                seq_choice = seq_names[sample_batch.train_id]
                sample_batch.train_id += 1
                if sample_batch.train_id >= len(seq_names):
                    sample_batch.EPOCH_FLAG=True
                    sample_batch.train_id = 0
            else:
                seq_choice = seq_names[sample_batch.test_id]
                sample_batch.test_id += 1
                if sample_batch.test_id >= len(seq_names):
                    sample_batch.test_id = 0
            new_sample = None
            if type(seq_choice) is str and os.path.isfile(seq_choice):
                new_sample = read_npz(seq_choice, args)
            else:
                assert args.pre_load
                new_sample = seq_choice
            if new_sample is not None:
                break
        samples.append(new_sample)
        sample_names.append(new_sample['name'])
        sample_batch.num_vertices = max(
            sample_batch.num_vertices, samples[-1]['V'].shape[0])
        sample_batch.num_faces = max(
            sample_batch.num_faces, samples[-1]['F'].shape[0])

    inputs = torch.zeros(
        args.batch_size,
        sample_batch.num_vertices,
        sample_batch.input_features)
    targets = None

    mask = torch.zeros(args.batch_size, sample_batch.num_vertices, 1)
    vert_faces = []
    laplacian = []
    diracs = [[], []]


    for b, sam in enumerate(samples):
        num_vertices, input_channel = sam['input'].shape

        inputs[b, : num_vertices, : input_channel] = sam['input']

        target_dist = sam['target_dist']
        vert_faces.append((sam['V'], sam['F']))

        if targets is None:
            sample_batch.output_features = target_dist.size(1)
            targets = torch.zeros(args.batch_size, sample_batch.num_vertices, sample_batch.output_features)

        mask[b, : num_vertices] = 1
        targets[b, :target_dist.shape[0], :sample_batch.output_features] = target_dist

        if 'mass' in sam:
            mass[b,:num_vertices,0] = sam['mass']

        if 'L' in sam:
            L = sam['L']
            laplacian.append(L)
        if 'Di' in sam:
            diracs[0].append(sam['Di'])
            diracs[1].append(sam['DiA'])
    device = torch.device('cuda' if args.cuda else 'cpu')

    laplacian = util.sparse_diag_cat(laplacian, sample_batch.num_vertices, sample_batch.num_vertices)

    if diracs[0] != []:
        diracs = [util.sparse_diag_cat(diracs[0], 4*sample_batch.num_faces, 4*sample_batch.num_vertices).to(device),
                   util.sparse_diag_cat(diracs[1], 4*sample_batch.num_vertices, 4*sample_batch.num_faces).to(device)]
    if len(laplacian) > 0:
        surface_operator = laplacian
    else:
        surface_operator = diracs
    if torch.is_tensor(surface_operator):
        surface_operator = surface_operator.to(device)
    mask = mask.to(device)

    return inputs.to(device), targets.to(device), mask,  surface_operator, vert_faces, sample_names, None
