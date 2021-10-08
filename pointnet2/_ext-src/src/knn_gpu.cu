#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3) radius_ref(b,n)
// output: idx(b, n, nsample), dist(b, n, nsample)
__global__ void knn_kernel(int b, int n, int m, int nsample, 
        const float *__restrict__ radius_ref,
        const float *__restrict__ xyz, const float *__restrict__ new_xyz,
        int *__restrict__ idx, float *__restrict__ dists) {
  int batch_index = blockIdx.x;
  new_xyz += batch_index * m * 3;
  xyz += batch_index * n * 3;
  idx += n * nsample * batch_index;
  dists += n * nsample * batch_index;
  radius_ref += n * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int j = index; j < n; j += stride) {
    float x = xyz[j * 3 + 0];
    float y = xyz[j * 3 + 1];
    float z = xyz[j * 3 + 2];
    float radius = radius_ref[j];
    float radius2 = radius * radius;
    float d_max = 1e6;
    int i_max = -1;
    int idx_base = j * nsample;
    int cnt = 0;

    for (int k = 0; k < m; ++k) {
      float new_x = new_xyz[k * 3 + 0];
      float new_y = new_xyz[k * 3 + 1];
      float new_z = new_xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      //if (d2 < radius2 && d2 < d_max) {
      if (d2 < d_max) {
        if (cnt < nsample) {
            idx[idx_base + cnt] = k;
            dists[idx_base + cnt] = d2;
            if (cnt+1 == nsample) {
                int i_max2 = 0;
                float d_max2 = 0;
                for (int l = 0; l < nsample; ++l)
                {
                    if (d_max2 < dists[idx_base + l])
                    {
                        i_max2 = l;
                        d_max2 = dists[idx_base + l];
                    }
                }
                d_max = d_max2;
                i_max = i_max2;
            }
        }
        else {
            int i_max2 = 0;
            float d_max2 = 0;
            idx[idx_base + i_max] = k;
            dists[idx_base + i_max] = d2;

            for (int l = 0; l < nsample; ++l)
            {
                if (d_max2 < dists[idx_base + l])
                {
                    i_max2 = l;
                    d_max2 = dists[idx_base + l];
                }
            }
            d_max = d_max2;
            i_max = i_max2;
        }
        ++cnt;
      }
    }
  }
}

void knn_kernel_wrapper(int b, int n, int m, int nsample, 
        const float *radius_ref,
        const float *xyz, const float *new_xyz, 
        int *idx, float* dists) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  knn_kernel<<<b, opt_n_threads(n), 0, stream>>>(
      b, n, m, nsample, radius_ref, xyz, new_xyz, idx, dists);
  CUDA_CHECK_ERRORS();
}


