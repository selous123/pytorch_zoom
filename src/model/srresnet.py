import torch
import torch.nn as nn
import math
from model import common

def make_model(args, parent=False):
    return _NetG(args)

class _Residual_Block(nn.Module):
    def __init__(self, n_feats):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(n_feats, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(n_feats, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output

class _NetG(nn.Module):
    def __init__(self, args):
        super(_NetG, self).__init__()

        conv = common.default_conv
        scale = args.scale[0]
        if args.n_colors == 4:
            scale = scale * 2
        self.n_feats = args.n_feats

        self.conv_input = nn.Conv2d(in_channels=args.n_colors, out_channels=args.n_feats, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, args.n_resblocks)

        self.conv_mid = nn.Conv2d(in_channels=args.n_feats, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(args.n_feats, affine=True)

        # define tail module
        self.upscale = common.Upsampler(conv, scale, self.n_feats, act=False)


        # self.upscale4x = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PixelShuffle(2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PixelShuffle(2),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        self.conv_output = nn.Conv2d(in_channels=self.n_feats, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.n_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale(out)
        out = self.conv_output(out)
        return out
