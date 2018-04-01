'''
This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
This Source Code Form is subject to the terms of the Mozilla Public License 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.
'''

import cupy
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple
import time
import torch
import numpy as np
from string import Template
from utils.cuda.batch_csr import BatchCSR

Stream = namedtuple('Stream', ['ptr'])

class SparseBMM(object):

    def __init__(self):
        self.cached_functions = {}

    def __call__(self, values, col_ind, col_ptr, size, dense):
        func_id = (size[0], size[1], size[2], dense.size(1), dense.size(2))
        if func_id in self.cached_functions:
            func = self.cached_functions[func_id]
        else:
            kernel = open('utils/cuda/sparse_bmm.cu', 'r').read()
            kernel = Template(kernel).substitute(BATCH_SIZE=size[0],
                                                 SPARSE_NUM_ROWS=size[1],
                                                 SPARSE_NUM_COLS=size[2],
                                                 DENSE_NUM_ROWS=dense.size(1),
                                                 DENSE_NUM_COLS=dense.size(2))

            program = Program(kernel, 'sparse_bmm.cu')
            ptx = program.compile()

            m = function.Module()
            m.load(bytes(ptx.encode()))

            func = m.get_function('sparse_bmm')
            self.cached_functions[func_id] = func

        values = values.contiguous()
        col_ind = col_ind.contiguous()
        col_ptr = col_ptr.contiguous()
        dense = dense.contiguous()
        result = values.new(size[0], size[1], dense.size(2))
        block = (8, 8, 8)

        grid = tuple([(result.size(i) + block[i] - 1) // block[i] for i in range(3)])
        func(grid=grid, block=block,
             args=[result.data_ptr(), values.data_ptr(), col_ind.data_ptr(),
                   col_ptr.data_ptr(), dense.data_ptr()], stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return result

sparse_bmm = SparseBMM()

if __name__ == "__main__":
    tmp = torch.zeros(1).cuda()
    batch_csr = BatchCSR()
    sparse_bmm = SparseBMM()

    indices, values, size = np.load(open('utils/tmp_data/L_.pt', 'rb'))

    a_ = torch.sparse.FloatTensor(indices, values, size).cuda().transpose(2, 1)
    batch_size, num_nodes, num_faces = a_.size()

    a = a_.to_dense()

    for _ in range(10):
        b = torch.randn(batch_size, num_faces, 16).cuda()
        torch.cuda.synchronize()
        time1 = time.time()
        result = torch.bmm(a, b)
        torch.cuda.synchronize()
        time2 = time.time()
        print("{} CuBlas dense bmm".format(time2 - time1))

        torch.cuda.synchronize()
        time1 = time.time()
        col_ind, col_ptr = batch_csr(a_.indices(), a_.size())
        my_result = sparse_bmm(a_.values(), col_ind, col_ptr, a_.size(), b)
        torch.cuda.synchronize()
        time2 = time.time()
        print("{} My sparse bmm".format(time2 - time1))

        print("{} Diff".format((result-my_result).abs().max()))
