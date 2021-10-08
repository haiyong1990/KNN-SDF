#ifndef _NN_DEVOX_CUH
#define _NN_DEVOX_CUH

// CUDA function declarations
void nn_devoxelize(int b, int c, int n, int r, int r2, int r3,
                   bool is_training, const float *coords,
                   const float *feat, int *inds,
                   float *outs);
void nn_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                        const float *grad_y, float *grad_x);

#endif
