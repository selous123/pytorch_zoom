from model import common

import torch.nn as nn
import torch
import torch.nn.functional as F

# from torch.nn import MaxPool1D
# import functional as F

# class ChannelPool(MaxPool1D):
#     def forward(self, input):
#         n, c, w, h = input.size()
#         input = input.view(n,c,w*h).permute(0,2,1)
#         pooled =  F.max_pool1d(input, self.kernel_size, self.stride,
#                         self.padding, self.dilation, self.ceil_mode,
#                         self.return_indices)
#         _, _, c = input.size()
#         input = input.permute(0,2,1)
#         return input.view(n,c,w,h)


## Return model list consisting of zoom model and self-supervised learning model
def make_model(args, parent=False):
    return EDSR_Zoom(args)


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

class CAconvAttn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAconvAttn, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool1 = nn.AdaptiveAvgPool2d(64)
        # feature channel downscale and upscale --> channel weight
        self.conv1 = nn.Sequential(
                nn.Conv2d(channel, channel, 3, padding=1, dilation=2, bias=True),
                nn.ReLU(inplace=True),
        )

        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        # 64 * 256 * 256  -> 64 * 64 * 64
        y = self.avg_pool1(x)
        y = self.conv1(y) # 64 * 64 * 64

        y = self.avg_pool2(y) # 64 * 1 * 1
        y = self.conv2(y) # 64 * 1 * 1
        return x * y


## parameter-free Spatial Attention (SA) Layer
class PFSpatialAttn(nn.Module):

    """Spatial Attention Layer"""
    def __init__(self):
        super(PFSpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,256,256
        x = x.mean(1, keepdim=True)  # e.g. 32,1,256,256
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,256*256
        y = x
        for b in range(x.size(0)):
            y[b] /= torch.sum(y[b])
        y = y.view(x.size(0),1,h,w)
        return x * y

## spatial attention (SA) Layer
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class PatchNonlocalPool(nn.Module):
    def __init__(self, n_feats, patch_size=16):
        super(PatchNonlocalPool,self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(n_feats, n_feats,kernel_size=3,stride=1,padding=1,bias=False)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        b,n,h,w = x.shape
        kernel_size = self.patch_size
        kernel_stride = self.patch_size
        # [b*64, 16, 16, 16, 16]
        # x [b*64, 256, 256] => [b*64, 256, 16, 16]
        x = x.view(b*n, h,w)
        a = x.unfold(1, kernel_size, kernel_stride).unfold(2,kernel_size,kernel_stride)
        a = a.contiguous().view(a.size(0), -1, a.size(3), a.size(4))
        # => [b*64, 256, 16, 16]
        #[b*64,256,256_fm] a_i
        a1 = a.view(*a.shape[:2],-1)
        #[b*64,256_fm,256] a_j
        a2 = a1.permute((0,2,1))
        #[b*64,256,256] => f(x_i, x_j)
        f1 = torch.matmul(a1, a2)
        f_div_C = F.softmax(f1, dim=-1)
        #[b*64,256,1,1]
        y1 = self.avg_pool1(a)
        #[b*64,256,1]
        y1 = y1.squeeze(-1)
        #[b*64,256,256]
        #y2 = torch.mul(f1,y1)
        #[b*64,256,1]
        y2 = torch.matmul(f_div_C, y1)
        #y3 = self.avg_pool2(y2)
        #[b, 64, 16, 16]
        y2 = y2.contiguous().view(b,n, int(h/self.patch_size), int(w/self.patch_size))
        y2 = self.conv(y2)
        #[b,64,1,1]
        y2 = self.avg_pool1(y2)
        return y2

## Patch Non-local Pooling with Channel Attention (CA) Layer
class PatchNonLocalCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PatchNonLocalCALayer, self).__init__()
        # global average pooling: feature --> point
        self.patchnonlocal_pool = PatchNonlocalPool(channel)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.patchnonlocal_pool(x)
        y = self.conv_du(y)
        return x * y


class ADL(nn.Module):
    def __init__(self, channel, drop_rate=0.5, drop_thr=0.5):
        super(ADL, self).__init__()
        assert 0 <= drop_rate <= 1 and 0 <= drop_thr <= 1
        self.drop_rate = drop_rate
        self.drop_thr = drop_thr
        self.attention = None
        self.drop_mask = None

        self.compress = ChannelPool()
        self.spatial = BasicConv(8, 8, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.channel = channel

    def forward(self, x):
        b = x.size(0)
        # Generate self-attention map
        #attention = torch.mean(x, dim=1, keepdim=True)
        x_compress = self.compress(x)
        x_rot = torch.cat([torch.rot90(x_compress,i,dims=[2,3]) for i in range(4)],dim=1) # 8 * 256 * 256
        attention = self.spatial(x_rot) # 8 * 256 * 256
        self.attention = attention[0]

        # Generate importance map
        importance_map = torch.sigmoid(attention)

        # Generate drop mask
        max_val, _ = torch.max(attention.view(b, -1), dim=1, keepdim=True)
        thr_val = max_val * self.drop_thr
        thr_val = thr_val.view(b, 1, 1, 1).expand_as(attention)
        drop_mask = (attention < thr_val).float()
        self.drop_mask = drop_mask
        # Random selection
        #random_tensor = torch.rand([], dtype=torch.float32) +
        #binary_tensor = random_tensor.floor()
        binary_tensor = self.drop_rate
        selected_map = (1. - binary_tensor) * importance_map + binary_tensor * drop_mask # b * 8 * 256 * 256

        ## broadcasting
        selected_map = selected_map.repeat(1, int(self.channel / 8), 1, 1)
        # Spatial multiplication to input feature map
        output = x.mul(selected_map)
        return output

    def get_maps(self):
        return self.attention, self.drop_mask


class SpatialAttn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttn, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(8, 8, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.channel = channel
    def forward(self, x):
        x_compress = self.compress(x)

        x_rot = torch.cat([torch.rot90(x_compress,i,dims=[2,3]) for i in range(4)],dim=1) # 8 * 256 * 256

        x_out = self.spatial(x_rot)
        scale = torch.sigmoid(x_out) # broadcasting
        scale = scale.repeat(1, int(self.channel / 8), 1, 1)
        return x * scale

class MixAttn(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super(MixAttn, self).__init__()
        self.attn1 = CALayer(n_feats,reduction)
        self.attn2 = SpatialAttn(n_feats)
        #self.attn2 = ADL(n_feats)
    def forward(self, x):
        ## serial mix attention
        x = self.attn1(x)
        x = self.attn2(x)
        return x


class NonLocalAttn(nn.Module):
    def __init__(self):
        super(NonLocalAttn, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(8, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        #x_compress = torch.mean(x,1).unsqueeze(1) # 1 * 256 * 256
        ## non-local operator
        x_rot = torch.cat([torch.rot90(x_compress,i,dims=[2,3]) for i in range(4)],dim=2) # 4 * 256 * 256
        x_out = self.spatial(x_rot)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

#Attn = MixAttn
#from model.context_block import ContextBlock2d
Attn = MixAttn

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

class EDSR_Zoom(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_Zoom, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        #act = nn.ReLU6(True)
        act = nn.ReLU(True)
        #act = common.Relu_n(20)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.model_ssl = EDSR_SSL(args)
        self.CALayer_head = Attn(n_feats)
        self.CALayer_tail = Attn(n_feats)

        self.attn = args.attn
        self.attn_maps = []
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        if args.attn is True:
            m_body = [
                RCABlock(
                    conv, n_feats, kernel_size, args.reduction, act=act, res_scale=args.res_scale
                ) for _ in range(n_resblocks)
            ]

        else:
            m_body = [
                common.ResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                ) for _ in range(n_resblocks)
            ]


        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        # 'relu_10'
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act='relu'),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        ## s_label : [b, n_colors, H, W]
        ## y_res : [b, C, H, W]
        s_label, y1, y2 = self.model_ssl(x)

        x = self.sub_mean(x)
        x = self.head(x)

        self.attn_maps.append(x)
        ## feature fusion : fusion
        x = x - y2
        if self.attn:
            x = self.CALayer_head(x)

        res = self.body(x)
        ## feature fusion
        res = res + y1

        ## Channel Attention
        #y_w = self.CALayer(y_res)
        #res = y_w * res
        if self.attn:
            res = self.CALayer_tail(res)

        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x, s_label

    def get_attnmaps(self):
        return self.attn_maps


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR_SSL(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_SSL, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        #act = nn.ReLU6(True)
        act = nn.ReLU(True)
        #act = common.Relu_n(20)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        # 'relu_10'
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act='relu'),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        x1 = x

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x, x1, res
