#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : training.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 16.02.2020
# Last Modified Date: 20.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from pvcnn.modules.voxelization import Voxelization, voxelize
import pvcnn.modules.functional as CF
from pointnet2.utils import pointnet2_utils
import time
import math
import random


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, resolution=32, cfg=None):
        self.model = model
        self.evaluator = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.r = resolution
        self.cfg = cfg

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def parallel(self):
#        # self.model = torch.nn.DataParallel(self.model)
#        # print("traing on device: ", self.model.device_ids)
#        # self.model = self.model.to(self.device)
#
#        os.environ['MASTER_ADDR'] = 'localhost'
#        os.environ['MASTER_PORT'] = '%d'%(random.randint(1024,65535))
#        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
#        if torch.cuda.device_count() > 1:
#            torch.cuda.set_device(0)
#        self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
#        print("traing on device: ", self.model.device_ids)
        self.model = self.model.to(self.device)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        t_start  = time.time()
        device = self.device
        inputs = data.get('inputs').to(device)
        normals = data.get('inputs.normals').to(device)
        voxel_rec = None
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        self.model.train()
        model = self.model
        self.optimizer.zero_grad()
        kwargs = {"normals": normals}
        preds = model(points, inputs, occ, **kwargs)
        loss_dict = self.evaluator.compute_loss(points, inputs, [occ, voxel_rec], preds)
        loss_dict["loss_sum"].backward()
        self.optimizer.step()

        logits = preds[0][:, -1] # return last/finer results only.
        samples = {"points": points.detach(), "pred": logits.detach(),
                   "input": inputs.detach(), "gnd": occ.detach(),
                   }
        losses = {k:loss_dict[k].item() for k in loss_dict}
        metrics = self.evaluator.compute_metric([occ, voxel_rec], preds, None)

        return samples, losses, metrics

    def forward(self, points, inputs, occ, sample, **kwargs):
        M = 25000
        if "points_per_batch" in self.cfg["test"]:
            M = self.cfg["test"]["points_per_batch"]
        B, N, _ = points.size()
        for ii in range(int(math.ceil(N/1.0/M))):
            if occ is not None:
                occ_t = occ[:, ii*M:(ii+1)*M]
            else:
                occ_t = None
            preds = self.model(points[:, (ii*M):(ii+1)*M].contiguous(), inputs, occ_t, sample, **kwargs)
            if ii == 0:
                logits, q_z, occ_voxel = preds
            else:
                logits = torch.cat([logits, preds[0]], dim=2)
        return [logits, q_z, occ_voxel]

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold

        # Compute elbo
        inputs = data.get('inputs').to(device)
        normals = data.get('inputs.normals').to(device)

        B = inputs.shape[0]
        # occ_grid = data.get('voxels').to(device)
        voxel_rec = None # [B,1,D,H,W], r=32
        kwargs = {"normals": normals}

        ## random sampled points
        with torch.no_grad():
            points_rand = data.get('points_iou').to(device)
            occ_rand = data.get('points_iou.occ').to(device)
            N = 100000
            # N = 200000
            points_rand = points_rand[:,:N]
            occ_rand = occ_rand[:,:N]
            assert(points_rand.size()[1] == occ_rand.size()[1])
            preds_rand = self.forward(points_rand, inputs, None, sample=self.eval_sample, **kwargs)
            logits_rand = preds_rand[0][:,-1]
        metrics_rand = self.evaluator.compute_metric([occ_rand, voxel_rec], preds_rand, points_rand)
        metrics_rand = {k+"_rand":v for k,v in metrics_rand.items()}

        samples = {"points_rand": points_rand, "gnd_rand": occ_rand, "pred_rand": logits_rand}
        metrics = metrics_rand

        return samples, {}, metrics


