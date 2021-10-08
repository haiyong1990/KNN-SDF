#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 17.02.2020
# Last Modified Date: 19.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import warnings
warnings.simplefilter(action="ignore")
import GPUtil
import time
import argparse
#########################################################################
# setup CUDA_VISIBLE_DEVICES before importing pytorch
def setup_GPU(ngpu=1, xsize=10000):
    ## setup GPU
    ## detect and use the first available GPUs
    gpus = GPUtil.getGPUs()
    print("Detect available gpus: ")
    idxs = []
    mems = []
    counter = 0
    while len(idxs) == 0:
        for ii,gpu in enumerate(gpus):
            if gpu.memoryFree > xsize:
                idxs.append(ii)
                mems.append(gpu.memoryFree)
        if len(idxs) == 0:
            time.sleep(60)
            counter += 1
            if counter%(60*12) == 0:
                print("%d hours passed"%(counter/60))
    idxs = [v for _, v in sorted(zip(mems, idxs), reverse=True)]
    idxs = sorted(idxs[:ngpu])
    # idxs = [3,]
    GPU_IDS = ",".join([str(v) for v in idxs])
    print("Use Gpu, ", GPU_IDS)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    return list(range(len(idxs)))

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--restore', action='store_true', help='Restore network if available.')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpu to use .')
g_args = parser.parse_args()
g_gpu_ids = setup_GPU(g_args.ngpu)
#########################################################################

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import tensorflow as tf
import numpy as np
import matplotlib; matplotlib.use('Agg')
import glob
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
from im2mesh.net_optim import NetOptim
from im2mesh.utils.train_callbacks import (
    TensorboardLoggerCallback, TrainSaverCallback
)
from im2mesh.tf_log import setup_log
torch.backends.cudnn.enabled = False


def train():
    ## setup which layer you want to tune
    global g_args, g_gpu_ids
    args, gpu_ids = g_args, g_gpu_ids
    layers = "all"
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda:0" if is_cuda else "cpu")
    brestore = args.restore

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']*len(gpu_ids)
    batch_size_test = cfg['test']['batch_size']*len(gpu_ids)

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    elif not brestore:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    ## setup
    setup_log(os.path.join(out_dir, "sdf_model.log"))

    # save_configs and source codes
    import yaml
    import shutil
    cfg_dir = os.path.join(out_dir,"srcs")
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    # copy and save srcs
    fname = os.path.join(cfg_dir,"config.yaml")
    with open(fname, 'w') as ofile:
        yaml.dump(cfg, ofile, default_flow_style=False)

    for folder in ["onet", "encoder", "data"]:
        if not os.path.exists(cfg_dir + "/%s/"%folder):
            shutil.copytree("im2mesh/%s/"%folder, cfg_dir+"/%s/"%folder)

    for fpath in glob.glob("im2mesh/*.py"):
        shutil.copy(fpath, cfg_dir)
    shutil.copy("train.py", cfg_dir + "/train.py")

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg)
    # val_dataset = config.get_dataset('test', cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=6, shuffle=True,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_test*len(gpu_ids), num_workers=2, shuffle=False,
        collate_fn=data.collate_remove_none,
        worker_init_fn=data.worker_init_fn)

    # Model, optimizer, scheduler, checkpoint_saver, net_trainer,
    # step_trainer
    #  os.environ['MASTER_ADDR'] = 'localhost'
    #  os.environ['MASTER_PORT'] = '12355'
    #  torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    #  if len(gpu_ids) > 1:
    #      torch.cuda.set_device(gpu_ids[0])
    model = config.get_model(cfg, device=device, dataset=train_dataset)
    # model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpu_ids)
    optimizer, lr_scheduler = config.get_optimizer(cfg, model, layers)
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)
    net_optim = NetOptim(cfg, model, trainer, checkpoint_io)
    if brestore:
        net_optim.restore_model("model.pt")
    net_optim.print_net_params()
    if len(gpu_ids) > 1 and args.ngpu>1:
        torch.cuda.set_device(gpu_ids[0])
        trainer.parallel()

    ## callbacks for logging/visualization
    log_interval = int(cfg["training"]["visualize_every"])
    callbacks = [TensorboardLoggerCallback(out_dir),
                 TrainSaverCallback(cfg, out_dir, log_interval)]

    ## train/val
    tf.logging.info("Training network: ")
    net_optim.train(train_loader, val_loader, [optimizer, lr_scheduler], callbacks)

    #  ## TODO: test
    #  net_optim.eval(val_loader, callbacks)


## traing
train()
