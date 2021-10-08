import torch
import torch.nn as nn

g_b_align_corners = True
class UNet3DV2(nn.Module):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    Code credits: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Unet3D.py
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        global g_b_align_corners
        self.b_align_corners = g_b_align_corners
        self.in_dim = in_dim
        self.out_dim = out_dim
        h_dim = 128

        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.lrelu = self.act
        self.upsacle = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=self.b_align_corners)
        self.softmax = nn.Softmax(dim=1)

        # 1st level
        self.conv3d_c1_1 = nn.Conv3d(self.in_dim, h_dim, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(h_dim, h_dim, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(h_dim, h_dim)
        self.inorm3d_c1 = nn.BatchNorm3d(h_dim)

        # 2nd level
        self.conv3d_c2 = nn.Conv3d(h_dim, h_dim * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(h_dim * 2, h_dim * 2)
        self.inorm3d_c2 = nn.BatchNorm3d(h_dim * 2)

        # 3rd level
        self.conv3d_c3 = nn.Conv3d(h_dim * 2, h_dim * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(h_dim * 4, h_dim * 4)
        self.inorm3d_c3 = nn.BatchNorm3d(h_dim * 4)

        # 4th level
        self.conv3d_c4 = nn.Conv3d(h_dim * 4, h_dim * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(h_dim * 8, h_dim * 8)
        self.inorm3d_c4 = nn.BatchNorm3d(h_dim * 8)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(h_dim * 8, h_dim * 4)

        # >> up-4th level
        self.conv3d_l1 = nn.Conv3d(h_dim * 4, h_dim * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l1 = nn.BatchNorm3d(h_dim * 4)

        # up-3th level
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(h_dim * 8, h_dim * 8)
        self.conv3d_l2 = nn.Conv3d(h_dim * 8, h_dim * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(h_dim * 4,
                                                                                             h_dim * 2)

        # up-2 level
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(h_dim * 4, h_dim * 4)
        self.conv3d_l3 = nn.Conv3d(h_dim * 4, h_dim * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(h_dim * 2,
                                                                                             h_dim)
        # up-1 level
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(h_dim * 2, h_dim * 2)
        self.conv3d_l4 = nn.Conv3d(h_dim * 2, self.out_dim, kernel_size=1, stride=1, padding=0,
                                   bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(feat_out),
            self.act)

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            self.act,
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            self.act,
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            self.act,
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=self.b_align_corners),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(feat_out),
            self.act)

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        ds1 = out

        # up-level-1 localization pathway
        out = self.conv3d_l1(out)
        out = self.inorm3d_l1(out)
        out = self.lrelu(out)

        # Level 2 localization pathway
        # print(out.shape)
        # print(context_3.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        return [out_pred, ds3, ds2, ds1]



