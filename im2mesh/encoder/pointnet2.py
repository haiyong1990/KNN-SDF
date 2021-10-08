import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import (
    PointnetSAModule, PointnetFPModule, PointnetSAModuleMSG
)


class PointNet2(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, **kwargs):
        super().__init__()
        use_xyz = True

        in_dim = 32
        self.fc_in = nn.Sequential(
            pt_utils.Conv1d(dim, in_dim, bn=True),
        )

        self.SA_modules = nn.ModuleList()
        c_in = in_dim
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                # radii=[0.05, 0.1],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + in_dim, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.fc_pt = nn.Sequential(
            pt_utils.Conv1d(128, hidden_dim, bn=True),
        )
        self.fc_g = nn.Sequential(
            pt_utils.Conv1d(c_out_3, hidden_dim, bn=True),
        )

    def forward(self, pointcloud: torch.cuda.FloatTensor, **kwargs):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns:
            ---------
            feat_pt: [B,N,F]
            feat_g: [B,F]
        """
        xyz = pointcloud # [B,N,3]
        features = self.fc_in(pointcloud.transpose(1,2)) # [B,F,N]

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            # print(l_xyz[i].size(), l_features[i].size() if l_features[i] is not None else "")
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        feat_pt = self.fc_pt(l_features[0]).transpose(1, 2).contiguous() # [B,N,F]
        feat_g = torch.max(self.fc_g(li_features), dim=-1)[0] # [B,F]
        return feat_g, [feat_pt], None, None

class PointNet2v2(PointNet2):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, **kwargs):
        super().__init__()
        use_xyz = True

        in_dim = 32
        self.fc_in = nn.Sequential(
            pt_utils.Conv1d(dim, in_dim, bn=True),
        )

        self.SA_modules = nn.ModuleList()
        c_in = in_dim
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.02, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.05, 0.15],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1, 0.3],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.5],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 256], [c_in, 256, 256, 256]],
                use_xyz=use_xyz
            )
        )
        c_out_3 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1,
                radii=[0.4, 1.0],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 256, 512]],
                use_xyz=use_xyz
            )
        )
        c_out_4 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + in_dim, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_2, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_4 + c_out_3, 512, 512]))



