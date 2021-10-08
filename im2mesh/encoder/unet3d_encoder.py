import torch
import torch.nn as nn

from pvcnn.modules.voxelization import Voxelization, voxelize
from im2mesh.encoder.pointnet import ResPNLayer

from im2mesh.encoder.unet3d_med import UNet3DV2


#################################################################################
## PVConv to extract voxels based features
class PVConvv2Encoder(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, resolution=32, nlevel=2, **kwargs):
        super().__init__()
        ## pointnet for feature extraction
        self.cfg = kwargs
        in_dim, out_dim = hidden_dim, c_dim
        self.pn_layer = ResPNLayer(in_dim, dim=dim, nlevel=nlevel)

        ## voxelization
        self.r = resolution
        self.voxelization = Voxelization(self.r, normalize=False, eps=1e-6)

        ## UNet3D.
        flag = False
        self.unet_conv = UNet3DV2(in_dim, out_dim)
        flag = True
        out_dim = 128

        ## final fc
        self.fc = nn.Sequential(
            # nn.Linear(out_dim*4, c_dim),
            nn.Linear(128*4, c_dim),
        )

    def forward(self, inputs, **kwargs):
        """
        inputs:
            @inputs: [B,N,3]
        outputs:
            @outputs: [B,N,F]
        """
        features = self.pn_layer(inputs).transpose(1,2).contiguous() #[B,F,N]
        coords = inputs.transpose(1,2).contiguous() # [B,3,N]
        voxel_features, voxel_coords = self.voxelization(features, coords)
        ms_features = self.unet_conv(voxel_features) # from dense to raw

        B,F,D,H,W = ms_features[-1].size()
        g_feat = torch.max(ms_features[-1].reshape(B,F,-1), dim=2)[0] # [B,F]
        g_feat = self.fc(g_feat) # [B,F]

        voxels = None
        basis_params = None
        return g_feat, ms_features, features, basis_params, voxels
           #[B,F], [[B,C,D,H,W]], [[B,N,3]], [B,1,D,H,W], [B,1,D,H,W]



