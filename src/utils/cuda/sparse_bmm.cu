// This file is part of source code for "Surface Networks", Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna. CVPR 2018.

// Copyright (C) 2018 Ilya Kostrikov <kostrikov@cs.nyu.edu> and Zhongshi Jiang <zhongshi@cims.nyu.edu>
 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#define BATCH_SIZE ${BATCH_SIZE}
#define SPARSE_NUM_ROWS ${SPARSE_NUM_ROWS}
#define SPARSE_NUM_COLS ${SPARSE_NUM_COLS}
#define DENSE_NUM_ROWS ${DENSE_NUM_ROWS}
#define DENSE_NUM_COLS ${DENSE_NUM_COLS}

extern "C"
__global__ void sparse_bmm(float *C, const float *values,
                           const long *col_ind, const long *col_ptr,
                           const float *dense) {
        long b = blockIdx.x * blockDim.x + threadIdx.x;
        long i = blockIdx.y * blockDim.y + threadIdx.y;
        long j = blockIdx.z * blockDim.z + threadIdx.z;

        if (!(b < BATCH_SIZE && i < SPARSE_NUM_ROWS)) {
          return;
        }

        __shared__ long s_col_ptr[8][8][8];

        if (threadIdx.z < 2) {
          s_col_ptr[threadIdx.x][threadIdx.y][threadIdx.z] = col_ptr[b * (SPARSE_NUM_ROWS + 1) + i + threadIdx.z];
        }

        __syncthreads();

        float value = 0.0f;
        long start = s_col_ptr[threadIdx.x][threadIdx.y][0];
        long end = s_col_ptr[threadIdx.x][threadIdx.y][1];

        __shared__ long s_col_ind[8][8][8];
        __shared__ float s_values[8][8][8];

        for (long k = start; k < end; k += 8) {
          __syncthreads();

          s_col_ind[threadIdx.x][threadIdx.y][threadIdx.z] = col_ind[k + threadIdx.z];
          s_values[threadIdx.x][threadIdx.y][threadIdx.z] = values[k + threadIdx.z];

          __syncthreads();

          for (long kt = 0; kt < 8 && k + kt < end; ++kt) {
            long col_id = s_col_ind[threadIdx.x][threadIdx.y][kt];
            value += s_values[threadIdx.x][threadIdx.y][kt] * dense[b * DENSE_NUM_ROWS * DENSE_NUM_COLS + col_id * DENSE_NUM_COLS + j];
            //long col_id = col_ind[k + kt];
            //value += values[k + kt] * dense[b * DENSE_NUM_ROWS * DENSE_NUM_COLS + col_id * DENSE_NUM_COLS + j];
          }
        }

        if (b < BATCH_SIZE && i < SPARSE_NUM_ROWS && j < DENSE_NUM_COLS) {
          C[b * SPARSE_NUM_ROWS * DENSE_NUM_COLS + i * DENSE_NUM_COLS + j] = value;
        }
}
