from model import common

import torch.nn as nn


def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.n_colors = args.n_colors
        scale = args.scale[0]
        if self.n_colors == 4:
            scale = scale * 2
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


        self.attn =  args.attn

        # define head module
        #print(n_feats)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        if self.attn is True:
            m_body = [
                common.RCABlock(
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
        #print(scale)
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        self.feats = []
        if self.n_colors == 3:
            x = self.sub_mean(x)
        x = self.head(x)

        self.feats.append(x)
        res = self.body(x)
        res += x

        x = self.tail(res)
        if self.n_colors == 3:
            x = self.add_mean(x)
        # print(self.get_attnmaps())
        return x

    def get_attnmaps(self):
        return self.feats

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
