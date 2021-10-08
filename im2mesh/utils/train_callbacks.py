#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : train_callbacks.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 19.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import torch
import tensorflow as tf
from tensorboardX import SummaryWriter
import numpy as np
from im2mesh.utils import visualize as vis
from im2mesh.utils.io import export_pointcloud

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def add_dict(self, writer, val_dict, prefix, epoch_id, val_type="scalar"):
        for k,v in val_dict.items():
            if val_type=="scalar":
                if isinstance(v, (int, float)):
                    writer.add_scalar(prefix+"/"+k, v, epoch_id)
            elif val_type=="hist":
                writer.add_histogram(prefix+"/"+k, v, epoch_id)


class TensorboardLoggerCallback(Callback):
    def __init__(self, BASE_DIR):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to add valuable
            information to the tensorboard logs such as the losses
            and accuracies
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = os.path.join(BASE_DIR, "logs")
        if not os.path.exists(self.path_to_files):
            os.makedirs(self.path_to_files)
        self.loss = 1e10

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return
        epoch_id = kwargs['epoch_id']

        ## add loss for tensorboard visualization
        self.writer = SummaryWriter(self.path_to_files)
        for name in ["train_loss", "val_loss"]:
            self.add_dict(self.writer, kwargs[name], 'data/%s'%(name), epoch_id)
        self.add_dict(self.writer, {"lr": kwargs['lr']}, 'data/learning_rate', epoch_id)
        lr = kwargs['lr']
#        if "train_sample" in kwargs and epoch_id>0:
#            sample = kwargs["train_sample"]
#            for k in sample:
#                if k.endswith("weight"):
#                    if "param_" in k:
#                        self.add_dict(self.writer, {k:lr*sample[k]}, 'param/', epoch_id, "hist")
#                    if "update_" in k:
#                        self.add_dict(self.writer, {k:lr*sample[k]}, 'update/', epoch_id, "hist")
        self.writer.close()
        #  ## save best model according to validatio loss terms.
        #  if "val_loss" in kwargs and self.loss > kwargs["val_loss"]["metric"]:
        #      self.loss = self.loss > kwargs["val_loss"]["metric"]
        #      net = kwargs['net']
        #      torch.save(net.state_dict(), os.path.join(self.path_to_files,"model_best.pth"))



class TrainSaverCallback:
    def __init__(self, cfg, BASE_DIR, log_interval=10, max2save=50):
        self.cfg = cfg
        self.log_interval = log_interval
        self.max2save = max2save
        self.im_path = os.path.join(BASE_DIR, "images")
        self.shape_path = os.path.join(BASE_DIR, "shapes")
        for f in [self.im_path, self.shape_path]:
            if not os.path.exists(f):
                os.makedirs(f)

    def __call__(self, *args, **kwargs):
        """ Save Input/Target/Predict/Diff, Corner/Line/Poly_maps """
        if kwargs['step_name'] != "epoch":
            return
        epoch = 1
        if 'epoch_id' in kwargs:
            epoch = kwargs['epoch_id']
        if epoch%self.log_interval!= 0:
            return

        ## load cfgs
        in_type = self.cfg["data"]["input_type"]
        th = self.cfg["test"]["threshold"]

        ## save the meshes
        for split in ["val", "test"]:
            if split+"_sample" not in kwargs \
                    or len(kwargs[split + "_sample"]) == 0:
                continue
            sample = kwargs[split + "_sample"]
            sample_input = sample["input"].data.cpu().numpy()
            sample_pred = (sample["pred"]>=0.0).data.cpu().numpy()
            sample_gnd = (sample["gnd"]>=0.5).data.cpu().numpy()
            points = (sample["points"]).data.cpu().numpy()
            if "pred_voxel" in sample:
                voxel_pred = (sample["pred_voxel"]>=0.0).float().data.cpu().numpy()
                voxel_gnd = (sample["gnd_voxel"]).data.cpu().numpy()
            nmax = min(self.max2save, sample_input.shape[0])

            ## output 2d images
            for ii in range(nmax):
#                input_img_path = os.path.join(
#                    self.im_path, 'sample%03d_%s_%03d_in.png' % (epoch,split,ii) )
#                vis.visualize_data(sample_input[ii], in_type, input_img_path)
#
#                points_pred = points[ii][np.nonzero(sample_pred[ii]), :].reshape((-1,3))
#                pred_img_path = os.path.join(
#                    self.im_path, 'sample%03d_%s_%03d_grid_pred.png' % (epoch,split,ii) )
#                vis.visualize_pointcloud( points_pred,out_file= pred_img_path )
#
#                pred_img_path = os.path.join(
#                    self.im_path, 'sample%03d_%s_%03d_grid_pred_layer.png' % (epoch,split,ii) )
#                vis.visualize_pointcloud_layer( sample["points"][ii], sample["pred"][ii], out_file= pred_img_path )
#
#                points_gnd = points[ii][np.nonzero(sample_gnd[ii]), :].reshape((-1,3))
#                gnd_img_path = os.path.join(
#                    self.im_path, 'sample%03d_%s_%03d_grid_gnd.png' % (epoch,split,ii) )
#                vis.visualize_pointcloud( points_gnd, out_file=gnd_img_path )
#
#                pred_eq_gnd = points[ii][np.nonzero(sample_gnd[ii]*sample_pred[ii]), :].reshape((-1,3))
#                pred_ge_gnd = points[ii][np.nonzero(sample_gnd[ii]<sample_pred[ii]), :].reshape((-1,3))
#                pred_le_gnd = points[ii][np.nonzero(sample_gnd[ii]>sample_pred[ii]), :].reshape((-1,3))
#                pred_img_path = os.path.join(
#                    self.im_path, 'sample%03d_%s_%03d_grid_diff.png' % (epoch,split,ii) )
#                vis.visualize_pointcloud_diff(pred_eq_gnd, pred_ge_gnd, pred_le_gnd, out_file=pred_img_path)


#                if "pred_voxel" in sample:
#                    pred_img_path = os.path.join(
#                        self.im_path, 'sample%03d_%s_%03d_voxel_pred.png' % (epoch,split,ii) )
#                    vis.visualize_voxels(voxel_pred[ii], pred_img_path)
#                    gnd_img_path = os.path.join(
#                        self.im_path, 'sample%03d_%s_%03d_voxel_gnd.png' % (epoch,split,ii) )
#                    vis.visualize_voxels(voxel_gnd[ii], gnd_img_path)
#                    diff_img_path = os.path.join(
#                        self.im_path, 'sample%03d_%s_%03d_voxel_diff.png' % (epoch,split,ii) )
#                    vis.visualize_voxel_diff(voxel_pred[ii], voxel_gnd[ii], diff_img_path)


                if "pred_rand" in sample:
                    points_rand = sample["points_rand"][ii]
                    gnd_rand = sample["gnd_rand"][ii] >= 0.5
                    pred_rand = sample["pred_rand"][ii] >= 0.0
                    intersect_rand = torch.index_select(points_rand, 0, torch.nonzero(pred_rand*gnd_rand).reshape(-1))
                    pred_ge_gnd_rand = torch.index_select(points_rand, 0, torch.nonzero(pred_rand>gnd_rand).reshape(-1))
                    gnd_ge_pred_rand = torch.index_select(points_rand, 0, torch.nonzero(pred_rand<gnd_rand).reshape(-1))

                    pred_img_path = os.path.join(
                        self.im_path, 'sample%03d_%s_%03d_rand_diff.png' % (epoch,split,ii) )
                    vis.visualize_pointcloud_diff(
                            intersect_rand.data.cpu().numpy(),
                            pred_ge_gnd_rand.data.cpu().numpy(),
                            gnd_ge_pred_rand.data.cpu().numpy(),
                            out_file=pred_img_path)

                    # gnd_rand = torch.masked_select(points_rand, gnd_rand).reshape(-1, 3)
                    gnd_rand = torch.index_select(points_rand, 0, torch.nonzero(gnd_rand).reshape(-1)).reshape(-1, 3)
                    gnd_rand = gnd_rand.data.cpu().numpy()
                    gnd_img_path = os.path.join(
                        self.im_path, 'sample%03d_%s_%03d_rand_gnd.png' % (epoch,split,ii) )
                    vis.visualize_pointcloud(gnd_rand, out_file=gnd_img_path, color='g')

                    pred_rand = torch.index_select(points_rand, 0, torch.nonzero(pred_rand).reshape(-1)).reshape(-1, 3)
                    pred_rand = pred_rand.data.cpu().numpy()
                    pred_img_path = os.path.join(
                        self.im_path, 'sample%03d_%s_%03d_rand_pred.png' % (epoch,split,ii) )
                    vis.visualize_pointcloud(pred_rand, out_file=pred_img_path, color='r')

                    #pred_img_path = os.path.join(
                    #    self.im_path, 'sample%03d_%s_%03d_rand_pred_layer.png' % (epoch,split,ii) )
                    #vis.visualize_pointcloud_layer( sample["points_rand"][ii], sample["pred_rand"][ii], out_file= pred_img_path )


