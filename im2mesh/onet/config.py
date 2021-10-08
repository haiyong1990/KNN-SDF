import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    resolution = cfg['model']['resolution']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    decoder_kwargs["resolution"] = resolution
    decoder_kwargs["cfg"] = cfg
    encoder_kwargs["resolution"] = resolution
    encoder_kwargs["cfg"] = cfg

    if "hidden_dim" in encoder_kwargs:
        decoder_kwargs["eh_dim"] = encoder_kwargs["hidden_dim"]
    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )

    if z_dim != 0:
        encoder_latent = models.encoder_latent_dict[encoder_latent](
            dim=dim, z_dim=z_dim, c_dim=c_dim,
            **encoder_latent_kwargs
        )
        p0_z = get_prior_z(cfg, device)
    else:
        encoder_latent = None
        p0_z = None

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, **encoder_kwargs
        )
    else:
        encoder = None

    # decoder = decoder.to("cuda:0")
    model = models.OccupancyNetwork(
        decoder, encoder, encoder_latent, p0_z, device=device, cfg=cfg,
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        resolution=cfg['model']['resolution'],
        cfg=cfg
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.
    ==> im2mesh/config.py, get_generator

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)
    if True:
        generator = generation.Generator3D(
            model,
            device=device,
            points_batch_size=cfg['generation']['batch_size'],
            threshold=cfg['test']['threshold'],
            resolution0=cfg['generation']['resolution_0'],
            upsampling_steps=cfg['generation']['upsampling_steps'],
            sample=cfg['generation']['use_sampling'],
            refinement_step=cfg['generation']['refinement_step'],
            simplify_nfaces=cfg['generation']['simplify_nfaces'],
            preprocessor=preprocessor,
        )

    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.
    ==> im2mesh/onet/config.py, get_model

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields. (setup corresponding files and configs.)
    ==> im2mesh/config.py, get_dataset

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    ## points in space
    fields['points'] = data.PointsField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
        cfg=cfg,
    )

#    ## return inputs with normal and without noise.
#    with_transforms = cfg['data']['with_transforms']
#
#    fields["inputs_raw"] = data.PointCloudField(
#        cfg['data']['pointcloud_file'],
#        data.SubsamplePointcloud(cfg['data']['points_subsample']//4),
#        with_transforms=with_transforms,
#    )

    if mode in ('val', 'test'):
    # if True:
        ## points in space
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
                cfg=cfg,
            )

#    voxels_file = cfg['data']['voxels_file']
#    if voxels_file is not None:
#        fields['voxels'] = data.VoxelsField(voxels_file)

    return fields

