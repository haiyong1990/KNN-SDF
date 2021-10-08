#include "ball_query.h"
#include "utils.h"


void knn_kernel_wrapper(int b, int n, int m, int nsample, 
        const float *radius_ref,
        const float *xyz, const float *new_xyz, 
        int *idx, float* dists) ;


at::Tensor knn(at::Tensor xyz, at::Tensor new_xyz, 
        at::Tensor radius_ref, const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(radius_ref);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(radius_ref);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({xyz.size(0), xyz.size(1), nsample},
                   at::device(xyz.device()).dtype(at::ScalarType::Int));
  at::Tensor dists =
      torch::zeros({xyz.size(0), xyz.size(1), nsample},
                   at::device(xyz.device()).dtype(at::ScalarType::Float));

  if (new_xyz.type().is_cuda()) {
    knn_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1), nsample, 
            radius_ref.data<float>(), xyz.data<float>(), new_xyz.data<float>(), 
            idx.data<int>(), dists.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return idx;
}
