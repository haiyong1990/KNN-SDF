#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : __init__.py
# Author            : Occ-Net
# Date              : 16.02.2020
# Last Modified Date: 20.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from im2mesh.common import compute_iou, compute_iou_tensor
import pvcnn.modules.functional as CF
from pvcnn.modules.voxelization import voxelize
from chamferdist import ChamferDistance
chamferDist = ChamferDistance()


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
        cfg: configurations.
    '''

    def __init__(self, decoder, encoder=None, encoder_latent=None, p0_z=None,
                 device=None, cfg=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.cfg= cfg
        self.loss_cfg = {}
        if "loss" in cfg["model"]:
            self.loss_cfg = self.cfg["model"]["loss"]
        self.thresh = cfg["test"]["threshold"]

        print(device)
        self.decoder = decoder.to(device)

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z

    def forward(self, p, inputs, occ=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points [B,N1,3]
            inputs (tensor): conditioning input [B,N2,3]
            sample (bool): whether to sample for z

        Returns:
            logits (a list of logits or sdf): [B,X,N1]
            q_z (prior distribution): ?
            c (voxel representation): [B,1,W,H,D]
        '''
        batch_size = p.size(0)
        # this could be global feature or (global feature, local features, local xyzs) depending on which encoder is used.
        c = self.encoder(inputs)
        assert(len(c) >= 4)
        if occ is None:
            z = self.get_z_from_prior((batch_size,), sample=sample)
            q_z = self.p0_z
        else:
            q_z = self.infer_z(p, occ, c[0], **kwargs)
            z = q_z.rsample()
        logits = self.decoder(p, z, c, inputs=inputs, **kwargs)
        # return [logits, q_z, c[-1]]
        return [logits, torch.zeros((batch_size,), device=inputs.device), c[-1]]

    def cvt_occ2border(self, points, occ):
        B,N,_ = points.size()
        grouper = pointnet2_utils.QueryAndGroup(0.05, 6, use_xyz=False)
        occ_knn = grouper(
            points, points, occ.reshape(B,1,N).float() #[B,N,3], [B,N,3], [B,C,N]
        )  # (B, C, npoint, nsample)
        th = 0.3
        mask1 = (torch.mean(occ_knn, dim=3).reshape(B,N,1)>=th)
        mask2 = (torch.mean(occ_knn, dim=3).reshape(B,N,1)<=1-th)
        return mask1 * mask2

    def compute_metric(self, out_gnd, out_pred, points=None, **kwargs):
        ''' Computes metrics, including sdf.iou and voxel.iou

        Args:
            out_gnd (a list):
                occ (tensor): occupancy values for points [B,N1]
                voxel_rec (tensor): occupancy values for voxels in given resolutions[B,1,D,H,W]
            out_pred (a list):
                logits (a list of tensor): estimated logits or sdf [B,N1]
                q_z (tensor): distribution priors.
                voxel_logits (tensor): predicted voxel occupancy
        '''
        occ, voxel_rec = out_gnd
        logits, _, voxel_logits = out_pred

        ## sdf.volume.iou
        # Compute iou: iou in range [0,1], logits in range [-inf, inf]
        occ_iou_np = (occ >= 0.5).float()
        occ_iou_hat_np = (nn.Sigmoid()(logits[:,-1]) >= self.thresh).float()
        iou_points = compute_iou_tensor(occ_iou_np, occ_iou_hat_np).mean().item()

        metrics = {'iou_points': iou_points}
        return metrics

    def dilate_occ(self, occ):
        conv = torch.nn.Conv3d(1, 1, 3, padding=1).to(occ.device)
        torch.nn.init.constant_(conv.weight, 1.0)
        occ = (conv(occ)/27.0 > 0.2).float()
        return occ

    def mse_loss(self, logits, occs, weights=1.0):
#        eps = 1e-6
#        mask = (torch.abs(occs-0.5) < eps).float()
#        loss_surf = torch.sum(nn.MSELoss(reduction='none')(
#                        logits, 0.0) * mask * weights, dim=-1).mean()
#        loss_space = torch.sum(nn.MSELoss(reduction='none')(
#                        logits, occs - 0.5) * (1-mask) * weights, dim=-1).mean()

        loss = torch.sum(nn.MSELoss(reduction='none')(
                        logits, occs - 0.5) * weights, dim=1).mean()
        return loss

    def compute_loss(self, points, inputs, out_gnd, out_pred, **kwargs):
        ''' Computes the expectation lower bound.

        Args:
            points (tensor): sampled grid points in 3D space [B,N1,3]
            inputs (tensor): sampled point clouds [B,N2,3]
            out_gnd (a list):
                occ (tensor): occupancy values for points [B,N1]
                voxel_rec (tensor): occupancy values for voxels in given resolutions[B,1,D,H,W]
            out_pred (a list):
                logits (a list of tensor): estimated logits or sdf [B,N1]
                q_z (tensor): distribution priors.
                voxel_logits (tensor): predicted voxel occupancy
        '''
        loss_dict = {}
        occ, voxel_rec = out_gnd
        logits, q_z, voxel_logits = out_pred
        # th = self.cfg["test"]["threshold"]
        th = 5.0
        occ = torch.unsqueeze(occ, dim=1).expand_as(logits)
        with torch.no_grad():
            w1 = 0.5/logits.size()[1]
            # w1 = 1.0/logits.size()[1]
            # w1 = 0.2
            weights = torch.ones((1,logits.size()[1]-1,1),
                    dtype=logits.dtype, device=logits.device) * w1
        w = 0.0 if "rec_error_w" not in self.loss_cfg else self.loss_cfg["rec_error_w"]
        if "rec_error" not in self.loss_cfg or\
                self.loss_cfg["rec_error"] == "cross_entropy":
            occ = occ * 0.9 + (1-occ)*0.1 # label smoothing
            if logits.size()[1] > 1:
                loss1 = F.binary_cross_entropy_with_logits(
                        logits[:,:-1], occ[:,:-1], weight=weights, reduction='none').mean()
            else:
                loss1 = 0.0
            loss2 = F.binary_cross_entropy_with_logits(
                    logits[:,-1:], occ[:,-1:], reduction='none').mean()
            loss_dict["loss_rec_ce"] =  (loss2 + loss1) * w
        elif self.loss_cfg["rec_error"] == "mse":
            if logits.size()[1] > 1:
                loss1 = self.mse_loss(logits[:,:-1], occ[:,:-1], weights)
            else:
                loss1 = 0.0
            loss2 = self.mse_loss(logits[:,-1:], occ[:,-1:])
            loss_dict["loss_rec_mse"] =  (loss2 + loss1) * w

        ## TODO: JS or waserstain ?
        if "kl_w" in self.loss_cfg and self.loss_cfg["kl_w"]>0.0:
            loss_dict["loss_kl"] = self.loss_cfg["kl_w"]*dist.kl_divergence(q_z, self.p0_z).sum(dim=-1).mean()
        loss_sum = 0.0
        for k in loss_dict:
            if k!= "loss_sum":
                ## assert takes a lot of time
                # assert (loss_dict[k].item() != loss_dict[k].item())
                loss_sum = loss_sum + loss_dict[k]
        loss_dict["loss_sum"] = loss_sum
        return loss_dict

    def compute_input_loss(self, normals, out_pred):
        ''' Computes the expectation lower bound.

        Args:
            normals (tensor): sampled point clouds [B,N1,3]
            out_pred (a list):
                logits (a list of tensor): estimated logits or sdf [B,N1]
                q_z (tensor): distribution priors.
                voxel_logits (tensor): predicted voxel occupancy
        '''
        loss_dict = {}
        logits = out_pred[0][:,-1,:]
        w = 0.0 if "loss_normal_w" not in self.loss_cfg else self.loss_cfg["loss_normal_w"]
        B = logits.size(0)
        logits = logits.reshape(B,-1,7)
        logits_ref = logits[:,:,-1]
        logits_koff = logits[:,:,:-1].reshape(B, -1, 3, 2)
        normals_pred = torch.sum(logits_koff, dim=-1) - 2*logits_ref.unsqueeze(dim=-1) # [B,N,3]
        normals_pred = normals_pred / torch.clamp(torch.norm(normals_pred, dim=-1, keepdim=True), min=1e-6)
        loss_normal = w * torch.mean(torch.abs(torch.sum(normals_pred*normals, dim=-1) - 1.0))

        loss_dict = {"loss_sum": loss_normal, "loss_normal": loss_normal}
        return loss_dict


    def infer_z(self, p, occ, c, **kwargs):
        ''' Infers z.
        TODO: you may want to explore multi-scale features in c.

        Args:
            p (tensor): points tensor [B, N1, 3]
            occ (tensor): occupancy values for occ [B,N1]
            c (tensor): latent conditioned code c [B,F]
        '''
        ## learning a random latents to encode shape variations.
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        if sample:
            z = self.p0_z.sample(size).to(self._device)
        else:
            z = self.p0_z.mean.to(self._device)
            z = z.expand(*size, *z.size())

        return z

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

