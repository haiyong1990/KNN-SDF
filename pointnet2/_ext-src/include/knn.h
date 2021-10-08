#pragma once
#include <torch/extension.h>

at::Tensor knn(at::Tensor xyz, at::Tensor new_xyz, at::Tensor radius_ref, const int nsample);
