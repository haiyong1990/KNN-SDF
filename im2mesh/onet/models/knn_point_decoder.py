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
from im2mesh.layers import ResnetBlockFC
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
        # f_pt = f_pt.expand_as(f_knn)

        # qk = torch.matmul(f_knn.permute(0,2,3,1), f_pt.permute(0,2,1,3))/np.sqrt(f_knn.size(1)) # [B,N,K,1]
        qk = torch.sum(f_knn*f_pt, dim=1, keepdim=True).permute(0,2,3,1)/np.sqrt(f_knn.size(1)) # [B,N,K,1]
        qk = qk.permute(0,3,1,2) # [B,1,N,K]
        return nn.Softmax(dim=-1)(qk)


class KNNPNSDFDecoder(nn.Module):
    ''' Joint KNN feature and point neighbor for decoder, try to reduce memory usage.
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
            .conv2d(c_dim, bn=True, activation=act, preact=False)\

        self.knn_mlp = pt_utils.Seq(eh_dim+c_dim*2)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.point_mlp = pt_utils.Seq(c_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.block_net = pt_utils.Seq(2*c_dim+3)\
            .conv2d(c_dim, bn=True, activation=act)\
            .conv2d(1, bn=False, activation=None, preact=False)#[B,F,N]

        self.knn_weighter = MutualAttentionV2(c_dim, c_dim, c_dim)


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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        # encode local point feature
        pdiff = (p.transpose(1,2).unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff).reshape(B,-1,N*K).contiguous()

        # predict sdf
        feat_g = feat_g.unsqueeze(dim=-1).expand_as(pfeat_knn)
        vfeat_knn = torch.cat([pfeat_knn, vfeat_pdiff, feat_g], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]
        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]
        net_interp = self.block_net(
                torch.cat([
                    vfeat_knn.reshape(B,-1,N,K),
                    vfeat_pts.reshape(B,-1,N,1).repeat(1,1,1,K),
                    pdiff.reshape(B,-1,N,K),
                    ], dim=1)) # [B,1,N,K]

        # attention to extract the most important features
        vweights = self.knn_weighter(
            vfeat_knn.reshape(B,-1, N, K),
            vfeat_pts.reshape(B,-1, N, 1),
        ) #[B,1,N,K]
        net_g = torch.sum(net_interp*vweights, dim=-1) #[B,1,N]
        return net_g


class KNNPNDecoder(nn.Module):
    ''' Point based KNN sdf interpolation.
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
            .conv2d(c_dim, bn=True, activation=act, preact=False)\

        self.knn_mlp = pt_utils.Seq(eh_dim+c_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.point_mlp = pt_utils.Seq(c_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

        self.block_net = pt_utils.Seq(2*c_dim)\
            .conv1d(c_dim, bn=True, activation=act)\
            .conv1d(1, bn=False, activation=None, preact=False)#[B,F,N]

        self.knn_weighter = MutualAttentionV2(c_dim, c_dim, c_dim)

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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        # encode local point feature
        pdiff = (p.transpose(1,2).unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff).reshape(B,-1,N*K).contiguous()

        # predict sdf
        vfeat_knn = torch.cat([pfeat_knn, vfeat_pdiff], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]
        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]

        # attention to extract the most important features
        vweights = self.knn_weighter(
            vfeat_knn.reshape(B,-1, N, K),
            vfeat_pts.reshape(B,-1, N, 1),
        ) #[B,1,N,K]
        vfeat_knn = torch.sum(vweights * vfeat_knn.reshape(B,-1,N,K), dim=-1) #[B,F,N]

        net_g = self.block_net(
                torch.cat([vfeat_knn, vfeat_pts], dim=1)) # [B,1,N]
        return net_g


class KNNPNnofusionDecoder(KNNPNDecoder):
    ''' KNN based SDF interpolation by interpolated features only
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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]
        net_g = self.block_net(vfeat_pts) # [B,1,N]
        return net_g


class KNNPNDistwDecoder(KNNPNDecoder):
    ''' KNN based SDF interpolation using distance weights
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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        # encode local point feature
        pdiff = (p.transpose(1,2).unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff).reshape(B,-1,N*K).contiguous()

        # predict sdf
        vfeat_knn = torch.cat([pfeat_knn, vfeat_pdiff], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]
        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]

        # attention to extract the most important features
        with torch.no_grad():
            vweights = 1.0/ torch.clamp(torch.norm(pdiff, dim=1, keepdim=True), min=1e-6)
            vweights = vweights/ torch.sum(vweights, dim=-1, keepdim=True)
        vfeat_knn = torch.sum(vweights * vfeat_knn.reshape(B,-1,N,K), dim=-1) #[B,F,N]

        net_g = self.block_net(
                torch.cat([vfeat_knn, vfeat_pts], dim=1)) # [B,1,N]
        return net_g


class KNNPNPoolwDecoder(KNNPNDecoder):
    ''' KNN based SDF interpolation using global pooling
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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        # encode local point feature
        pdiff = (p.transpose(1,2).unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff).reshape(B,-1,N*K).contiguous()

        # predict sdf
        vfeat_knn = torch.cat([pfeat_knn, vfeat_pdiff], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]
        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]

        # attention to extract the most important features
        vfeat_knn = torch.max(vfeat_knn.reshape(B,-1,N,K), dim=-1)[0]  #[B,F,N]

        net_g = self.block_net(
                torch.cat([vfeat_knn, vfeat_pts], dim=1)) # [B,1,N]
        return net_g


class KNNPNNoPDiffDecoder(KNNPNDecoder):
    ''' Ours - pdiff
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
        self.knn_mlp = pt_utils.Seq(eh_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        vfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        # predict sdf
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]
        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]

        # attention to extract the most important features
        vweights = self.knn_weighter(
            vfeat_knn.reshape(B,-1, N, K),
            vfeat_pts.reshape(B,-1, N, 1),
        ) #[B,1,N,K]
        vfeat_knn = vfeat_knn.reshape(B,-1, N, K)
        vfeat_knn = torch.sum(vweights * vfeat_knn, dim=-1) #[B,F,N]

        net_g = self.block_net(
                torch.cat([vfeat_knn, vfeat_pts], dim=1)) # [B,1,N]
        return net_g


class KNNPNnoknnFeatDecoder(KNNPNDecoder):
    '''  Ours - knnfeat
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
        self.knn_mlp = pt_utils.Seq(c_dim)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)\
            .conv1d(c_dim, bn=True, activation=act, preact=False)#[B,F,N]

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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get points features
        with torch.no_grad():
            dist, idx = pn2_utils.three_nn(p, inputs)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
        vfeat_pts = pn2_utils.three_interpolate(
                feat_pt, idx, weight
        )#[B,F,N]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]

        # encode local point feature
        pdiff = (p.transpose(1,2).unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_knn = self.pdiff_mlp(pdiff).reshape(B,-1,N*K).contiguous()

        # predict sdf
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]
        vfeat_pts = self.point_mlp(vfeat_pts)  # [B,F,N*1]

        # attention to extract the most important features
        vweights = self.knn_weighter(
            vfeat_knn.reshape(B,-1, N, K),
            vfeat_pts.reshape(B,-1, N, 1),
        ) #[B,1,N,K]
        vfeat_knn = torch.sum(vweights * vfeat_knn.reshape(B,-1,N,K), dim=-1) #[B,F,N]

        net_g = self.block_net(
                torch.cat([vfeat_knn, vfeat_pts], dim=1)) # [B,1,N]
        return net_g


class KNNPNnointerpDecoder(KNNPNDecoder):
    ''' ours -interp.
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
        p = p.contiguous()
        inputs = inputs.contiguous()
        B, N, _= p.size()

        feat_g, feat_list = c[:2] #[B, 512], [[B,M,F]] , [[B,M,3]], [B,C,D,H,W]
        feat_pt = feat_list[0].transpose(1,2).contiguous() # [B,F,M]

        # get neighboring features
        K = self.cfg["model"]["K"]
        with torch.no_grad():
            radius = 100*torch.ones((B,N), dtype=torch.float, device=p.device)
            idxs = pn2_utils.KNN_query(K, radius, p, inputs) # [B,N,K]
        points_knn = pn2_utils.grouping_operation(inputs.transpose(1,2).contiguous(), idxs) # [B,C,N,K]
        pfeat_knn = pn2_utils.grouping_operation(feat_pt, idxs).reshape(B,-1,N*K)  # [B,C,N*K]

        # encode local point feature
        pdiff = (p.transpose(1,2).unsqueeze(dim=-1) - points_knn) # [B,3,N,K]
        vfeat_pdiff = self.pdiff_mlp(pdiff).reshape(B,-1,N*K).contiguous()

        # predict sdf
        vfeat_knn = torch.cat([pfeat_knn, vfeat_pdiff], dim=1)
        vfeat_knn = self.knn_mlp(vfeat_knn) # [B,F,N*K]

        # attention to extract the most important features
        vweights = self.knn_weighter(
            vfeat_knn.reshape(B,-1, N, K),
        ) #[B,1,N,K]
        vweights = torch.softmax(vweights, dim=-1)
        vfeat_knn = torch.sum(vweights * vfeat_knn.reshape(B,-1,N,K), dim=-1) #[B,F,N]

        net_g = self.block_net(vfeat_knn) # [B,1,N]
        return net_g


