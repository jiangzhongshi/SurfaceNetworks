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

Stream = namedtuple('Stream', ['ptr'])


class BatchCSR(object):

    def __init__(self):
        self.cached_functions = {}

    def __call__(self, indices, size):
        if size in self.cached_functions:
            func = self.cached_functions[size]
        else:
            kernel = open('utils/cuda/batch_csr.cu', 'r').read()
            kernel = Template(kernel).substitute(BATCH_SIZE=size[0], NUM_ROWS=size[1])

            program = Program(kernel, 'batch_csr.cu')
            ptx = program.compile()

            m = function.Module()
            m.load(bytes(ptx.encode()))

            func = m.get_function('batch_csr')
            self.cached_functions[size] = func

        indices = indices.contiguous()
        
        col_ind = indices.new(indices.size(1))
        col_ptr = indices.new(size[0], size[1] + 1)
        col_ptr.fill_(indices.size(1))
        col_ptr.fill_(0)

        grid = ((indices.size(1) + 1024 - 1) // 1024, 1, 1)
        block = (1024, 1, 1)
        func(grid=grid, block=block,
             args=[indices.data_ptr(), col_ind.data_ptr(),
                   col_ptr.data_ptr(), indices.size(1)],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        #col_ptr[0:-1, -1] = col_ptr[1:, 0]
        return col_ind, col_ptr

batch_csr = BatchCSR()

if __name__ == "__main__":
    tmp = torch.zeros(1).cuda()
    batch_csr = BatchCSR()

    indices, values, size = np.load(open('../tmp_data/Di_.pt', 'rb'))
    a_ = torch.sparse.FloatTensor(indices, values, size).cuda()
    a_t = a_.transpose(2, 1).contiguous()
    print(a_)
    print(a_t)
    print((a_.indices() - a_t.indices()).abs().max())
