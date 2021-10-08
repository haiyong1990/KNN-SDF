import torch
import torch.nn as nn
import numpy as np

import pvcnn.modules.functional as F

__all__ = ['Voxelization', 'voxelize', 'voxelize_pcd']


def voxelize_pcd(pcd, stride, accumulate=False):
    pcd = pcd + 0.5
    coord2index = lambda x: np.clip(np.floor(x*stride), 0.0, stride-1).astype(np.int32)
    indices = coord2index(pcd[:,0])*stride*stride \
        + coord2index(pcd[:,1])*stride + coord2index(pcd[:,2])
    voxels = np.zeros((stride*stride*stride,), dtype=np.int32)
    if accumulate:
        np.add.at(voxels, indices, 1.0)
        voxels = voxels >= 2.0
    else:
        voxels[indices] = 1.0
    return np.reshape(voxels, (stride, stride, stride))

def voxelize(coords, resolution=32, normalize=True, eps=1e-6):
    eps = 1e-6
    coords = coords.detach()
    # print(torch.min(coords).item(), torch.max(coords).item())
    if normalize:
        norm_coords = coords - coords.mean(2, keepdim=True)
        norm_coords = norm_coords / ( norm_coords.norm(dim=1, keepdim=True\
                        ).max(dim=2, keepdim=True)[0] * 2.0 + eps) + 0.5
        # norm_coords = (coords - (-0.6))/1.3
    else:
        norm_coords = coords/1.0 + 0.5 ## [input ranges in [-0.5, 0.5]*1.2], rescale to ensure it is in [0.0, 1.0]
    # print(torch.min(norm_coords).item(), torch.max(norm_coords).item())
    # assert(torch.min(norm_coords).item()>=0.0 and torch.max(norm_coords).item()<=1.0)
    norm_coords = torch.clamp(norm_coords * resolution, 0, resolution - 1)
    vox_coords = torch.round(norm_coords).to(torch.int32)
    return vox_coords, norm_coords


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        '''
        inputs:
            @coords: [B, 3, N], in x,y,z order
            @features: [B,F,N]
        returns:
            @result: [B,F,W,H,D], note the order
        '''
        vox_coords, norm_coords =  voxelize(coords, self.r, self.normalize, self.eps)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')

