#ifndef _NN_DEVOX_HPP
#define _NN_DEVOX_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> nn_devoxelize_forward(const int r,
                                              const bool is_training,
                                              const at::Tensor coords,
                                              const at::Tensor features);

at::Tensor nn_devoxelize_backward(const at::Tensor grad_y,
                                  const at::Tensor indices,
                                  const int r);

#endif
