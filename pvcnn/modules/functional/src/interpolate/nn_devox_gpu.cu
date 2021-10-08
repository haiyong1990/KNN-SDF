#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: nn devoxlization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r   : voxel resolution
    r2  : r ** 2
    r3  : r ** 3
    coords : the coordinates of points, FloatTensor[b, 3, n]
    feat   : features, FloatTensor[b, c, r3]
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    outs   : outputs, FloatTensor[b, c, n]
*/
__global__ void nn_devoxelize_kernel(int b, int c, int n, int r, int r2,
                                            int r3, bool is_training,
                                            const float *__restrict__ coords,
                                            const float *__restrict__ feat,
                                            int *__restrict__ inds,
                                            float *__restrict__ outs) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  inds += batch_index * n * 1;
  feat += batch_index * c * r3;
  outs += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    float x = coords[i];
    float y = coords[i + n];
    float z = coords[i + n + n];
    float x_r = roundf(x);
    float y_r = roundf(y);
    float z_r = roundf(z);

    int idx000 = x_r * r2 + y_r * r + z_r;
    if (is_training) {
      inds[i] = idx000;
    }

    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      outs[j * n + i] = feat[jr3 + idx000];
    }
  }
}

/*
  Function: nn devoxlization (backward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r3  : voxel cube size = voxel resolution ** 3
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    grad_y : grad outputs, FloatTensor[b, c, n]
    grad_x : grad inputs, FloatTensor[b, c, r3]
*/
__global__ void nn_devoxelize_grad_kernel(
    int b, int c, int n, int r3, const int *__restrict__ inds,
    const float *__restrict__ grad_y, float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * n * 1;
  grad_x += batch_index * c * r3;
  grad_y += batch_index * c * n;

  for (int i = index; i < n; i += stride) {
    int idx000 = inds[i];
    for (int j = 0; j < c; j++) {
      int jr3 = j * r3;
      float g = grad_y[j * n + i];
      atomicAdd(grad_x + jr3 + idx000, g);
    }
  }
}

void nn_devoxelize(int b, int c, int n, int r, int r2, int r3,
                          bool training, const float *coords, const float *feat,
                          int *inds, float *outs) {
  nn_devoxelize_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r, r2, r3, training, coords, feat, inds, outs);
  CUDA_CHECK_ERRORS();
}

void nn_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                               const float *grad_y, float *grad_x) {
  nn_devoxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(
      b, c, n, r3, inds, grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
