from model import common

import torch.nn as nn


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

## Residual Channel Attention Block (RCAB)
class RCABlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCABlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, reduction))

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
        self.CALayer_head = CALayer(n_feats)
        self.CALayer_tail = CALayer(n_feats)

        self.attn = args.attn

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        if args.attn is True:
            m_body = [
                common.RCABlock(
                    conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
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

        ## feature fusion : fusion
        x = x - y2
        if self.attn:
            x = self.CALayer_head(x)

        res = self.body(x)
        res += x

        ## Channel Attention
        #y_w = self.CALayer(y_res)
        #res = y_w * res

        ## feature fusion
        res = res + y1
        if self.attn:
            x = self.CALayer_tail(x)

        x = self.tail(res)
        x = self.add_mean(x)

        return x, s_label

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
