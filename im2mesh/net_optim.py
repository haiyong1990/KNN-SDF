#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : net_optim.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 19.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import torch.optim as optim
import tensorflow as tf
import copy
import numpy as np
import GPUtil
import time

## There may exist some confusion. This class may denote for the network training process. Maybe Trainer will be better.
class NetOptim(object):
    def __init__(self, cfg, net, trainer, ck):
        self.cfg = cfg
        self.net = net
        self.max_epochs = cfg["training"]["max_epochs"]
        self.epoch_it = 0
        self.mode = "train"
        self.ck = ck
        self.lr = self.cfg["training"]["lr"]
        self.loss_val_best = np.inf
        self.trainer = trainer

    def print_net_params(self):
        tf.logging.info("Network architecture: ")
        #  for k in self.net.state_dict:
        #      tf.logging.info("%s(%s)"%(k, str(self.net.state_dict[k])))
        tf.logging.info(self.net)
        nparameters = sum(p.numel() for p in self.net.parameters())
        tf.logging.info('Total number of parameters: %d' % nparameters)

    def restore_model(self, model_path):
        """ Restore a model parameters from the one given in argument

        params:
            @model_path: base filename
        """
        try:
            load_dict = self.ck.load(model_path)
        except FileExistsError:
            load_dict = dict()
        self.epoch_it = load_dict.get('epoch_it', -1)
        # self.lr = load_dict.get('lr', self.cfg["training"]["lr"])
        self.loss_val_best = load_dict.get('loss_val_best', np.inf)
        tf.logging.info("Restore model trained with epoch=%d, lr=%f, loss=%f"%(
            self.epoch_it, self.lr, self.loss_val_best)
        )

    ##////////////////////////////////////////////////////////////////
    ## set up metric updates
    def _update_metrics(self, metrics, loss, metric):
        """
        Update given losses and metrics. The update is in-place

        params:
            @metrics: a dict, the summary of metrics
            @loss: a dict, the present measured loss
            @metric, a dict or list the present measured metric.
        """
        for k,v in loss.items():
            metrics.setdefault(k, 0)
            metrics[k] = metrics[k] + loss[k]

        assert(isinstance(metric, dict))
        for k,v in metric.items():
            flag = k in metrics
            metrics.setdefault(k, v)

            if isinstance(v, dict):
                for k1 in v:
                    if k1 not in metrics[k]:
                        metrics[k][k1] = metric[k][k1]
                    else:
                        metrics[k][k1] += metric[k][k1]
            elif "_min" in k:
                metrics[k] = min(metric[k], metrics[k])
            elif "_max" in k:
                metrics[k] = max(metric[k], metrics[k])
            elif "_avg" in k or "_sum" in k:
                if flag:
                    metrics[k] = metric[k] + metrics[k]
            else:
                if flag:
                    metrics[k] = metric[k] + metrics[k]

        return metrics

    def _normalize_metrics(self, metrics, count):
        """
        Normalize the items in metrics with a size, nsample. The normalization is in-place.
        """
        for k in metrics:
            if "_max" in k or "_min" in k:
                metrics_ret[k] = metrics[k]
            elif "_avg" in k or k == "loss_sum":
                metrics[k] /= count
            else: ## average in default
                metrics[k] /= count
        return metrics

    ##////////////////////////////////////////////////////////////////
    ## set up optimizer
    def get_params(self, layers="all"):
        params = self.net.named_parameters()
        if layers == "all":
            params = {k:v for k,v in params}
        else:
            params = {k:v for k,v in params if k.startswith(layers)}
        tf.logging.info("Filter parameters (%s): "%layers)
        tf.logging.info(",".join(params.keys()))
        return params.values()

    def get_optimizer(self):
        if self.cfg["training"]["optimizer"] == "ADAM":
            return lambda x,y: optim.Adam(x, y)
        elif self.cfg["training"]["optimizer"] == "ADAMW":
            return lambda x,y: optim.AdamW(x, y)
        elif self.cfg["training"]["optimizer"] == "SGD":
            return lambda x,y: optim.SGD(x, y, momentum=0.9)
        else:
            raise "Unexpected optimizer: " + self.cfg["optm"]["optimizer"]

    def setup_optimizer(self, layers="all"):
        optimizer = self.get_optimizer()
        lr = self.lr
        if isinstance(layers, str):
            optimizer = self.get_optimizer()(self.get_params(layers=layers), lr)
        elif isinstance(layers, dict):
            param_list = []
            for name,lr in layers.items():
                param_list.append({"params": self.get_params(name), "lr": lr})
            optimizer = self.get_optimizer()(param_list, lr)

        lr_scheduler = None
        scheduler_name = self.cfg["training"]["scheduler"]
        scheduler_params = self.cfg["training"]["scheduler_params"]
        if scheduler_name == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', **scheduler_params)
        elif scheduler_name == "StepLR":
            lr_scheduler = StepLR(optimizer,
                                  scheduler_params["step_size"],
                                  gamma=scheduler_params["gamma"]
                                  )
        elif scheduler_name == "MultiStepLR":
            lr_scheduler = MultiStepLR(optimizer, **schedulr_params)
        return optimizer, lr_scheduler

    def get_largest_lr(self, opt):
        return np.max([v["lr"] for v in opt.param_groups])

    ##////////////////////////////////////////////////////////////////
    ## eval process
    def eval(self, test_loader, callbacks=None):
        """
            Trains the neural net
        Args:
            valid_loader (DataLoader): The Dataloader for testing
            callbacks (list): List of callbacks functions to call at each epoch
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        ## Run a train pass on the current epoch
        t = time.time()
        # Run the validation pass
        with torch.no_grad():
            test_loss, test_sample = self._validate_epoch(test_loader)
        torch.cuda.empty_cache()

        t_elapsed = time.time() - t

        # If there are callback call their __call__ method and pass in some arguments
        if callbacks:
            for cb in callbacks:
                cb(step_name="epoch", net=self.net,
                epoch_id=0, lr = 0.0,
                train_sample=[], val_sample=val_sample,
                train_loss= {}, val_loss= {},
                )
        tf.logging.info("/***********************************************/")
        tf.logging.info("Testing loss: /n" + str(states["val_metric"]) )
        tf.logging.info("Timing: %fs"%(t_elapsed/test_loader.size(0)) )

    ##////////////////////////////////////////////////////////////////
    ## train process
    def train(self, train_loader, valid_loader, optim_list, callbacks=None, **kwargs):
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
            optim_list (a list of optimizer and scheduler)
            callbacks (list): List of callbacks functions to call at each epoch
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        optimizer, lr_scheduler = optim_list

        ## restore learning rates.
        for ii in range(self.epoch_it):
            if lr_scheduler == None:
                pass
            elif self.cfg["training"]["scheduler"]=="ReduceLROnPlateau":
                lr_scheduler.step(val_loss["loss_sum"])
            else:
                lr_scheduler.step()
        lr = self.get_largest_lr(optimizer)
        states = {"lr": lr, "state_dict": [], "val_loss": self.loss_val_best}
        states["state_dict"] = copy.deepcopy(self.net.state_dict())
        tf.logging.info("Start with a learning rate of " + str(lr))

        identifier = self.cfg["model"]["identifier"]
        nepoch_good = 0
        for epoch in range(self.epoch_it+1, self.max_epochs):
            self.epoch_it = epoch
            tf.logging.info("\n\nEpoch#%d, learning rate#%e, model: %s"%(
                epoch, lr, identifier) )
            GPUtil.showUtilization()
            t = time.time()
            if not self.cfg["training"]["skip_training"]:
                train_loss, train_sample = self._train_epoch(
                    train_loader, optimizer, epoch)
            else:
                train_loss, train_sample = {"iou_points": 0.0}, {}
            tf.logging.info("Training Time: %d"%(time.time()-t))

            checkpoint_io = self.ck
            # Run the validation pass
            val_loss, val_sample = {}, {}
            if epoch%self.cfg["training"]["validate_every"] == 0:
                if "use_adaptive_sampling" in self.cfg["data"] and self.cfg["data"]["use_adaptive_sampling"]:
                    sample_ratio = self.cfg["data"]["sample_surf_ratio"]
                    self.cfg["data"]["sample_surf_ratio"] = 0.0

                with torch.no_grad():
                    val_loss, val_sample = self._validate_epoch(valid_loader)

                if "use_adaptive_sampling" in self.cfg["data"] and self.cfg["data"]["use_adaptive_sampling"]:
                    self.cfg["data"]["sample_surf_ratio"] = sample_ratio

                if states["val_loss"] >= -val_loss["iou_points_rand"]:
                    tf.logging.info("Update best model: %.3f, %.3f"%(
                        -val_loss["iou_points_rand"], states["val_loss"]))
                    states["val_loss"] = -val_loss["iou_points_rand"]
                    states["val_metric"] = val_loss
                    states["epoch"] = epoch
                    states["state_dict"] = copy.deepcopy(self.net.state_dict())
                    checkpoint_io.save('model_best.pt', epoch_it=epoch, lr=lr,
                                    loss_val_best=states["val_loss"])

            backup_every = self.cfg["training"]["backup_every"]
            if (backup_every > 0 and (epoch % backup_every) == 0):
                print('Backup checkpoint')
                checkpoint_io.save('model_%d.pt'%epoch, epoch_it=epoch, lr=lr,
                                loss_val_best=states["val_loss"])
            checkpoint_io.save('model.pt', epoch_it=epoch, lr=lr,
                            loss_val_best=states["val_loss"])

            # reschedule sample ratios.
            if "use_adaptive_sampling" in self.cfg["data"] and self.cfg["data"]["use_adaptive_sampling"]:
                if train_loss["iou_points"] >= 0.80:
                    nepoch_good += 1
                    if nepoch_good >= 10:
                        self.cfg["data"]["sample_surf_ratio"] = min(self.cfg["data"]["sample_surf_ratio"]+0.05, 0.6)
                        tf.logging.info("Increase surface points ratio: %.3f"%self.cfg["data"]["sample_surf_ratio"])
                        nepoch_good = 0

            # Reduce learning rate if needed
            if lr_scheduler == None:
                pass
            elif self.cfg["training"]["scheduler"]=="ReduceLROnPlateau":
                lr_scheduler.step(val_loss["loss_sum"])
            else:
                lr_scheduler.step()
                print("lr_scheduler step")
            lr_opt = self.get_largest_lr(optimizer)
            lr = lr_opt
            # if states["lr"] > lr_opt:
            #     tf.logging.info("Restore the best model if learning rate decreases.")
            #     self.net.load_state_dict(states["state_dict"])
            #     states["lr"] = lr_opt

            if callbacks:
                for cb in callbacks:
                    cb(step_name="epoch",
                    net=self.net,
                    train_sample=train_sample,
                    val_sample=val_sample,
                    epoch_id=epoch,
                    train_loss= train_loss,
                    val_loss=val_loss,
                    lr = lr_opt
                    )
            if lr_opt <= 5e-8:
                break

        if "epoch" in states:
            tf.logging.info("Best validation results at epoch#%d:"%states["epoch"])
            tf.logging.info("learning rate: %f"%states["lr"])
            tf.logging.info("val loss: " + str(states["val_metric"]))


    ## ////////////////////////////////////////////////////////////
    ## steps
    def _train_epoch(self, train_loader, optimizer, epoch):
        """
        returns:
            @metrics: evaluation metrics and loss terms
            @samples: a batch of examples
        """
        self.net.train()
        self.mode = "train"
        metrics = {}
        loss_names = {}
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
            t_start = time.time()
            ## adjust sampling for hard example training:
            samples, loss, metric = self.trainer.train_step(data)
            self._update_metrics(metrics, loss, metric)

            if batch_idx % self.cfg["training"]["print_per_batch"] == 0:
                tf.logging.info('Train Epoch: {} [{}*({:.0f}%), {}]'.format(
                    epoch, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), batch_idx
                ))
                tf.logging.info("Loss terms: " + ", ".join(["%s: %.4f"%(k,loss[k])
                                           for k in sorted(loss.keys())]))
            # print("train_step: %fs"%(time.time()-t_start))

        batch_idx += 1
        metrics = self._normalize_metrics(metrics, batch_idx)
        #      self._register_params(samples)
        tf.logging.info("Train set: ")
        tf.logging.info("Loss terms: " + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(loss.keys())]))
        tf.logging.info('Metrics: ' + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(metrics.keys())
            if k not in loss
        ]))
        return metrics, samples

    def _validate_epoch(self, val_loader, btest=False):
        self.net.eval()
        self.mode = "test"
        sample_list = {}
        loss = {}
        metrics = {}
        t_start = time.time()
        for batch_idx, data in enumerate(val_loader):
            # torch.backends.cudnn.enabled = False
            samples, loss, metric = self.trainer.eval_step(data)
            # torch.backends.cudnn.enabled = True
            self._update_metrics(metrics, loss, metric)
            if batch_idx == 0:
                sample_list = {k:samples[k].detach() for k in samples}
            # elif btest and batch_idx<20:
            elif batch_idx<20:
                sample_list = {
                    k: torch.cat([sample_list[k], samples[k].detach()], dim=0)
                    for k in samples
                }

        t_elapsed = time.time() - t_start
        batch_idx += 1
        metrics = self._normalize_metrics(metrics, batch_idx)
        tf.logging.info("Validation set: ")
        tf.logging.info("Loss terms: " + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(loss.keys())]))
        tf.logging.info('Metrics: ' + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(metrics.keys())
            if k not in loss
        ]))
        tf.logging.info('Time per batch: %fs.'%(t_elapsed/batch_idx))
        return metrics, sample_list


    ## ////////////////////////////////////////////////////////////
    ## debugs
    def _register_params(self, sample):
        state_dict = self.net.state_dict(keep_vars=True)
        for k in state_dict:
            if k.endswith("weight"):
                sample["param_"+k] = state_dict[k].data.cpu().numpy()
                sample["grad_"+k] = state_dict[k].grad.data.cpu().numpy()
                sample["update_"+k] = np.abs(sample["grad_"+k])/(1e-8+np.abs(sample["param_"+k]))


