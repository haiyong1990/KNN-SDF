from im2mesh.encoder import (
    pointnet, pointnet2, unet3d_encoder,
)


encoder_dict = {
    'pointnet_resnet': pointnet.ResnetPointnet,
    'pointnet2': pointnet2.PointNet2,
    'pvconv2': unet3d_encoder.PVConvv2Encoder,
}
