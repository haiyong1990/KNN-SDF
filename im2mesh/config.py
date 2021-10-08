#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : config.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 26.01.2020
# Last Modified Date: 20.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import yaml
from torch import optim
import tensorflow as tf
from torchvision import transforms
from im2mesh import data
from im2mesh import onet
from im2mesh import preprocess
from torch.optim.lr_scheduler import *


method_dict = {
    'onet': onet,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            if dict1[k] is None and v is not None:
               dict1[k] = dict()
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator

# Datasets loader
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']
    if mode in ["val", "test"] and "test_classes" in cfg['data']:
        categories = cfg['data']['test_classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    # print("testing shape3d")
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        # print(method, inputs_field)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        print("cats: ", categories)
        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg=cfg,
        )

    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']
    with_transforms = cfg['data']['with_transforms']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    elif input_type == 'pointcloud':
        tf_list = [
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ]
        if cfg['data']['pointcloud_outlier']:
            tf_list += [data.PointcloudOutlier()]
        if cfg['data']['pointcloud_hole']:
            tf_list += [data.PointcloudHole()]
        transform = transforms.Compose(tf_list)
        with_transforms = cfg['data']['with_transforms']

        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            with_transforms=with_transforms
        )
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field


def get_preprocessor(cfg, dataset=None, device=None):
    ''' Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    '''
    p_type = cfg['preprocessor']['type']
    cfg_path = cfg['preprocessor']['config']
    model_file = cfg['preprocessor']['model_file']

    if p_type == 'psgn':
        preprocessor = preprocess.PSGNPreprocessor(
            cfg_path=cfg_path,
            pointcloud_n=cfg['data']['pointcloud_n'],
            dataset=dataset,
            device=device,
            model_file=model_file,
        )
    elif p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor


def get_params(model, layers="all"):
    params = model.named_parameters()
    if layers == "all":
        params = {k:v for k,v in params}
    else:
        params = {k:v for k,v in params if k.startswith(layers)}
    tf.logging.info("Filter parameters (%s): "%layers)
    tf.logging.info(",".join(params.keys()))
    return params.values()


def get_optimizer(cfg, model, layers="all"):
    optimizer_func = None
    if cfg["training"]["optimizer"] == "ADAM":
        optimizer_func = lambda x,y: optim.Adam(x, y)
    elif cfg["training"]["optimizer"] == "ADAMW":
        optimizer_func = lambda x,y: optim.AdamW(x, y)
    elif cfg["training"]["optimizer"] == "SGD":
        optimizer_func = lambda x,y: optim.SGD(x, y, momentum=0.9)
    else:
        raise "Unexpected optimizer: " + cfg["training"]["optimizer"]

    ## optimizer
    lr = float(cfg["training"]["lr"])
    if isinstance(layers, str):
        optimizer = optimizer_func(get_params(model, layers), lr)
    elif isinstance(layers, dict):
        param_list = []
        for name,lr in layers.items():
            param_list.append({"params": get_params(model, name), "lr": lr})
        optimizer = optimizer_func(param_list, lr)

    ## scheduler
    lr_scheduler = None
    scheduler_name = cfg["training"]["scheduler"]
    scheduler_params = cfg["training"]["scheduler_params"]
    tf.logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    tf.logging.info("scheduler: " + str(scheduler_name))
    tf.logging.info("params: " + str(scheduler_params))
    tf.logging.info("optimizer: " + cfg["training"]["optimizer"])
    tf.logging.info("init lr = " + str(cfg["training"]["lr"]))
    tf.logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    if scheduler_name == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', **scheduler_params)
    elif scheduler_name == "StepLR":
        lr_scheduler = StepLR(optimizer,
                                scheduler_params["step_size"],
                                gamma=scheduler_params["gamma"]
                                )
    elif scheduler_name == "MultiStepLR":
        lr_scheduler = MultiStepLR(optimizer, **scheduler_params)
    return optimizer, lr_scheduler




