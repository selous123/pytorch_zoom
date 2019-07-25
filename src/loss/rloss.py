import torch.nn as nn
import torch

class RLoss(nn.Module):
    def __init__(self, args):
        super(RLoss, self).__init__()
        self.l_loss = nn.L1Loss()
        self.args = args


    def forward(self, fake, real):
        #fake : tuple (SR, F_Diff)
        #real : tuple (HR, Diff)
        sr, fake_diff = fake
        hr, diff = real

        #threshold = self.args.rloss_threshold * self.args.rgb_range / 255.0

        fake_diff = fake_diff / self.args.rgb_range
        diff = diff / self.args.rgb_range
        #print(fake_diff)
        #print(threshold)

        #fake_diff_byte = torch.gt(fake_diff, threshold).type(torch.cuda.FloatTensor)
        #diff_byte = torch.gt(diff, threshold).type(torch.cuda.FloatTensor)

        sr_activation = torch.mul(sr, diff)
        hr_activation = torch.mul(hr, diff)

        l1 = self.l_loss(sr_activation, hr_activation)

        sr_activation1 = torch.mul(sr, fake_diff)
        #hr_activation1 = torch.mul(hr, diff)

        l2 = self.l_loss(sr_activation1, hr_activation)

        loss = l1 + l2

        return loss
