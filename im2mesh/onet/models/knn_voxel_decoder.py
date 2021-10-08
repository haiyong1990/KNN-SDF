#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : lod_net.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 11.02.2020
# Last Modified Date: 16.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
import pvcnn.modules.functional as CF
from copy import deepcopy
import etw_pytorch_utils as pt_utils
import numpy as np
from pvcnn.modules.voxelization import voxelize
from pointnet2.utils import pointnet2_utils as pn2_utils # import KNN_query


class MutualAttentionV2(nn.Module):
    def __init__(self, dim_knn, dim_pdiff, dim_pt):
        super().__init__()
        act = nn.LeakyReLU(0.05, True)
        self.conv_knn = pt_utils.Seq(dim_knn)\
            .conv2d(128, bn=True, activation=act)
        self.conv_pdiff = pt_utils.Seq(dim_pdiff)\
            .conv2d(128, bn=True, activation=act)
        self.conv_pt = pt_utils.Seq(dim_pt)\
            .conv2d(128, bn=True, activation=act)

    def forward(self, f_knn, f_pt):
        """
        params:
            @f_pcd: [B,F,N,K], point cloud features for nn points
            @f_pt: [B,F,N,1], feature for testing points
        returns:
            @weights: [B,1,N,K]
        """
        f_knn = self.conv_knn(f_knn) # [B,F,N,K]
        f_pt  = self.conv_pt(f_pt) #[B,F,N,1]
        f_pt = f_pt.expand_as(f_knn)

        # qk = torch.matmul(f_knn.permute(0,2,3,1), f_pt.permute(0,2,1,3))/np.sqrt(f_knn.size(1)) # [B,N,K,1]
        qk = torch.sum(f_knn*f_pt, dim=1, keepdim=True).permute(0,2,3,1)/np.sqrt(f_knn.size(1)) # [B,N,K,1]
        qk = qk.permute(0,3,1,2) # [B,1,N,K]
        return nn.Softmax(dim=-1)(qk)


class KNN3DUNetDecoder(nn.Module):
    ''' Joint KNN decoder and 3DUNet.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): hidden dim of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''
    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_dim=256,
                 leaky=False, legacy=False, resolution=32, eh_dim=128, **kwargs):
        super().__init__()

        act = nn.LeakyReLU(0.05)
        self.z_dim = z_dim
        self.r = resolution

        self.cfg = kwargs["cfg"]

        self.pdiff_mlp = pt_utils.Seq(3)\
            .conv2d(c_dim, bn=True, activation=act, preact=False)\

        self.knn_mlp = pt_utils.Seq(c_dim+c_dim+c_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.interp_mlp = pt_utils.Seq(c_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.point_mlp = pt_utils.Seq(eh_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.block_net = pt_utils.Seq(c_dim*2)\
            .conv1d(c_dim, bn=True, activation=act)\
            .conv1d(1, bn=False, activation=None, preact=False)#[B,F,N]

        self.knn_weighter = pt_utils.Seq(c_dim)\
            .conv2d(c_dim, bn=True, activation=act)\
            .conv2d(1, bn=False, activation=None, preact=False)#[B,F,N]

    def forward(self, p, z, c, inputs=None, **kwargs):
        r"""
        inputs:
            @p: testing point set [B,N,3]
            @z: latent vector to control minor variations [B,F]
            @c: a list of g_feature/features/xyzs from inputs ([B,F], [B,M,F], [B,M,3])
            @inputs: input point cloud [B,M,3]
        outputs:
            @net: [B,1,N], multi-level sdf
        """
        p = p.transpose(1, 2) #[B, 3, N]
        B, F, N = p.size()

        feat_g, feat_list, feat_pt, _, voxels = c #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_out = feat_list[0] # [B, F, D, H, W]

        # get points features
        with torch.no_grad():
            i_coords, v_coords = voxelize(p, self.r, normalize=False)
        vfeat_pts = CF.trilinear_devoxelize(feat_out, v_coords, self.r, True) #[B,F,N]
        vfeat_pts = self.interp_mlp(vfeat_pts)

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p.transpose(1,2).contiguous(), inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        feat_pt = self.point_mlp(feat_pt)
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        with torch.no_grad():
            assert(points_knn.size() == (B,3,N,K))
            _, v_coords_knn = voxelize(
                    points_knn.reshape(B, 3, -1).contiguous(),
                    self.r,
                    normalize=False)
        vfeat_knn = CF.trilinear_devoxelize(feat_out, v_coords_knn, self.r, True) #[B,F,N*K]

        # encode local point feature
        pdiff = (p.unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff)

        # predict sdf
        vfeat_knn = torch.cat([pfeat_knn, vfeat_knn, vfeat_pdiff.reshape(B,-1,N*K)], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]

        # attention to extract the most important features
        vweights = self.knn_weighter(vfeat_knn.reshape(B,-1,N,K)) #[B,1,N,K]
        vweights = torch.softmax(vweights, dim=-1)
        vfeat = torch.sum(vfeat_knn.reshape(B,-1,N,K)*vweights, dim=-1)
        net_g = self.block_net(torch.cat([vfeat, vfeat_pts], dim=1)) # [B,1,N]

        return net_g


class KNN3DUNetNoFuseDecoder(KNN3DUNetDecoder):
    ''' Joint KNN decoder and 3DUNet.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): hidden dim of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''
    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_dim=256,
                 leaky=False, legacy=False, resolution=32, eh_dim=128, **kwargs):
        super().__init__(dim, z_dim, c_dim, hidden_dim, leaky, legacy, resolution, eh_dim, **kwargs)

        act = nn.LeakyReLU(0.05)
        self.block_net = pt_utils.Seq(c_dim)\
            .conv1d(c_dim, bn=True, activation=act)\
            .conv1d(1, bn=False, activation=None, preact=False)#[B,F,N]

    def forward(self, p, z, c, inputs=None, **kwargs):
        r"""
        inputs:
            @p: testing point set [B,N,3]
            @z: latent vector to control minor variations [B,F]
            @c: a list of g_feature/features/xyzs from inputs ([B,F], [B,M,F], [B,M,3])
            @inputs: input point cloud [B,M,3]
        outputs:
            @net: [B,1,N], multi-level sdf
        """
        p = p.transpose(1, 2) #[B, 3, N]
        B, F, N = p.size()

        feat_g, feat_list, feat_pt, _, voxels = c #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_out = feat_list[0] # [B, F, D, H, W]

        # get points features
        with torch.no_grad():
            i_coords, v_coords = voxelize(p, self.r, normalize=False)
        vfeat_pts = CF.trilinear_devoxelize(feat_out, v_coords, self.r, True) #[B,F,N]
        vfeat_pts = self.interp_mlp(vfeat_pts)

        net_g = self.block_net(vfeat_pts) # [B,1,N]

        return net_g


class KNN3DUNetNoInterpDecoder(KNN3DUNetDecoder):
    ''' Joint KNN decoder and 3DUNet.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_dim (int): hidden dim of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''
    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_dim=256,
                 leaky=False, legacy=False, resolution=32, eh_dim=128, **kwargs):
        super().__init__(dim, z_dim, c_dim, hidden_dim, leaky, legacy, resolution, eh_dim, **kwargs)

        act = nn.LeakyReLU(0.05)
        self.z_dim = z_dim
        self.r = resolution

        self.cfg = kwargs["cfg"]
        self.block_net = pt_utils.Seq(c_dim)\
            .conv1d(c_dim, bn=True, activation=act)\
            .conv1d(1, bn=False, activation=None, preact=False)#[B,F,N]

    def forward(self, p, z, c, inputs=None, **kwargs):
        r"""
        inputs:
            @p: testing point set [B,N,3]
            @z: latent vector to control minor variations [B,F]
            @c: a list of g_feature/features/xyzs from inputs ([B,F], [B,M,F], [B,M,3])
            @inputs: input point cloud [B,M,3]
        outputs:
            @net: [B,1,N], multi-level sdf
        """
        p = p.transpose(1, 2) #[B, 3, N]
        B, F, N = p.size()

        feat_g, feat_list, feat_pt, _, voxels = c #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_out = feat_list[0] # [B, F, D, H, W]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p.transpose(1,2).contiguous(), inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        feat_pt = self.point_mlp(feat_pt)
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        with torch.no_grad():
            assert(points_knn.size() == (B,3,N,K))
            _, v_coords_knn = voxelize(
                    points_knn.reshape(B, 3, -1).contiguous(),
                    self.r,
                    normalize=False)
        vfeat_knn = CF.trilinear_devoxelize(feat_out, v_coords_knn, self.r, True) #[B,F,N*K]

        # encode local point feature
        pdiff = (p.unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff)

        # predict sdf
        vfeat_knn = torch.cat([pfeat_knn, vfeat_knn, vfeat_pdiff.reshape(B,-1,N*K)], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]

        # attention to extract the most important features
        vweights = self.knn_weighter(vfeat_knn.reshape(B,-1,N,K)) #[B,1,N,K]
        vweights = torch.softmax(vweights, dim=-1)
        vfeat = torch.sum(vfeat_knn.reshape(B,-1,N,K)*vweights, dim=-1)
        net_g = self.block_net(vfeat) # [B,1,N]

        return net_g




