import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC
import time


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net_pt = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net_pt, dim=1)

        c = self.fc_c(self.actvn(net))

        # return c, [net_pt.transpose(1,2)], None, [p.transpose(1,2)]
        return c, [net_pt.transpose(1,2)], None, None


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, **kwargs):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, **kwargs):
        batch_size, N, D = p.size()
        # t_start = time.time()

        # output size: B x N X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net_F = self.block_4(net) # [B,N,F]

        # Recude to  B x F
        net = self.pool(net_F, dim=1)

        c = self.fc_c(self.actvn(net))
        # print("pnres: %fs"%(time.time() - t_start))

        # return c, [net_F], None, [p] #[B,F], [[B,N,F]], [[B,N,3]]
        return c, [net_F], None, None #[B,F], [[B,N,F]], [[B,N,3]]


class ResPNLayer(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        o_dim (int): output dim
        dim (int): input points dimension
    '''

    def __init__(self, out_dim=128, dim=3, nlevel=2):
        super().__init__()
        c_dim = out_dim//2
        self.fc_pos = nn.Linear(dim, 64)
        self.block_0 = ResnetBlockFC(64, c_dim)
        self.block_1 = ResnetBlockFC(2*c_dim, c_dim)
        self.blocks = nn.ModuleList()
        for ii in range(nlevel-2):
            self.blocks.append(ResnetBlockFC(2*c_dim, c_dim))
        self.pool = maxpool

    def forward(self, p):
        '''
        inputs:
            @p: [B,N,3] a list of points
        outputs:
            @net: [B,N,F]
        '''
        batch_size, N, D = p.size()

        # output size: B x N X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        for block in self.blocks:
            net = block(net)
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=2) #[B,N,F]

        return net #[B,N,F]

