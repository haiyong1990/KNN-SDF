#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : __init__.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 11.02.2020
# Last Modified Date: 16.02.2020
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
from im2mesh.onet.models import encoder_latent, decoder
from im2mesh.onet.models import knn_point_decoder as interp_decoder
from im2mesh.onet.models import knn_voxel_decoder
from im2mesh.onet.models.onet import OccupancyNetwork

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'cbatchnorm': decoder.DecoderCBatchNorm,
    "ours": interp_decoder.KNNPNDecoder,
    "ours_3dgrid": knn_voxel_decoder.KNN3DUNetDecoder,
}
