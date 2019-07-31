import math

import torch
import torch.nn as nn
import torch.nn.functional as F




## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        #self.spatial = nn.Conv2d(8, 8,kernel_size=7,stride=1,padding=(kernel_size-1) // 2,bias=False)
        self.channel = channel
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

## spatial attention (SA) Layer
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

import torch.nn.functional as F
class ChannelPool(nn.Module):
    def __init__(self, pool_size = 1):
        super(ChannelPool, self).__init__()
        #self.maxpool = nn.AdaptiveMaxPool1d(pool_size)
        self.avgpool = nn.AdaptiveAvgPool1d(pool_size)
        self.pool_size = pool_size
    def forward(self, x):
        n, c, w, h = x.shape
        x = x.view(n,c,w*h).permute(0,2,1) #n * hw * c
        #max_pooled = self.maxpool(x)
        avg_pooled = self.avgpool(x)

        #pooled = torch.cat( (max_pooled, avg_pooled), dim=-1 )
        pooled = avg_pooled
        _, _, c1 = pooled.shape
        pooled = pooled.permute(0,2,1).view(n,c1,w,h) # n * 2*pool_size * h * w
        return pooled



class SpatialAttn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttn, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(4, 8, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        #self.spatial = nn.Conv2d(8, 8,kernel_size=7,stride=1,padding=(kernel_size-1) // 2,bias=False)
        self.channel = channel
    def forward(self, x):
        x_compress = self.compress(x)

        x_rot = torch.cat([torch.rot90(x_compress,i,dims=[2,3]) for i in range(4)],dim=1) # 8 * 256 * 256

        x_out = self.spatial(x_rot)
        scale = torch.sigmoid(x_out) # broadcasting
        scale = scale.repeat(1, int(self.channel / 8), 1, 1)
        return x * scale

class MultiPoolingSpatialAttn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MultiPoolingSpatialAttn, self).__init__()
        kernel_size = 7
        self.compress1 = ChannelPool(pool_size=1) #=> 1 * H * W
        self.compress2 = ChannelPool(pool_size=2) #=> 2 * H * W
        self.compress3 = ChannelPool(pool_size=4) #=> 4 * H * W

        self.spatial1 = BasicConv(4, 8, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial2 = BasicConv(8, 8, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial3 = BasicConv(16, 8, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

        #self.spatial = nn.Conv2d(8, 8,kernel_size=7,stride=1,padding=(kernel_size-1) // 2,bias=False)
        self.channel = channel
    def forward(self, x):

        x_compress1 = self.compress1(x)
        x_compress2 = self.compress2(x)
        x_compress3 = self.compress3(x)

        x_rot1 = torch.cat([torch.rot90(x_compress1,i,dims=[2,3]) for i in range(4)],dim=1) # 4 * 256 * 256
        x_out1 = self.spatial1(x_rot1)

        x_rot2 = torch.cat([torch.rot90(x_compress2,i,dims=[2,3]) for i in range(4)],dim=1) # 8 * 256 * 256
        x_out2 = self.spatial2(x_rot2)

        x_rot3 = torch.cat([torch.rot90(x_compress3,i,dims=[2,3]) for i in range(4)],dim=1) # 16 * 256 * 256
        x_out3 = self.spatial3(x_rot3)

        x_out = x_out1 + x_out2 + x_out3
        scale = torch.sigmoid(x_out) # broadcasting
        scale = scale.repeat(1, int(self.channel / 8), 1, 1)
        x_scale = x * scale

        return x_scale

class MixAttnSA(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super(MixAttn, self).__init__()
        self.attn1 = CALayer(n_feats,reduction)
        self.attn2 = SpatialAttn(n_feats)
        #self.attn2 = ADL(n_feats)
        #self.attn2 = MultiPoolingSpatialAttn(n_feats)
    def forward(self, x):
        ## serial mix attention
        x = self.attn1(x)
        x = self.attn2(x)
        return x

class MixAttn(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super(MixAttn, self).__init__()
        self.attn1 = CALayer(n_feats,reduction)
        #self.attn2 = SpatialAttn(n_feats)
        #self.attn2 = ADL(n_feats)
        self.attn2 = MultiPoolingSpatialAttn(n_feats)
        #self.attn2 = SALayer(n_feats)
    def forward(self, x):
        ## serial mix attention
        x = self.attn1(x)
        x = self.attn2(x)
        return x

#Attn = MultiPoolingSpatialAttn
#Attn = SALayer
Attn = MixAttn
#Attn = CALayer
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


# class Relu_n(nn.Module):
#     def __init__(self, n):
#         super(Relu_n, self).__init__()
#         self.Num = n
#
#     def forward(self,x):
#         max = torch.zeros(x.shape).cuda()
#         min = torch.ones(x.shape).cuda()
#
#         min = min * self.Num
#         max_x = torch.max(x, max)
#         min_x = torch.min(max_x, min)
#         return min_x

class Relu_n(nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, n=6, inplace=True):
        super(Relu_n, self).__init__(0., n, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## Residual Channel Attention Block (RCAB)
class RCABlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCABlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(Attn(n_feats, reduction))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == False:
                    continue
                elif act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act.find('relu_')>-1:
                    n = int(act.split('_')[1])
                    m.append(Relu_n(n))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act.find('relu_')>-1:
                n = int(act.split('_')[1])
                m.append(Relu_n(n))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
