
import torch.nn as nn
import torch
def pairwiseFunc(images, edges, n = 16):
    """
    Argument:
        images: with shape [N, C, H, W]
        edges : with shape [N, C, H, W]
    Return:
        loss
    """
    #a = torch.randn(1, 1, 16, 16)
    #a = torch.from_numpy(np.arange(0,256)).view(1,1,16,16)
    #print(a)
    h,w = images.shape[2:]
    kernel_size = n
    kernel_stride = n
    # images [n,c,h,w] ==> [n, h/n * w/n, c, h/n, w/n]
    # edges [n,c,h,w] ==> [n, h/n * w/n, c,  h/n, w/n]

    #assert h % n == 0 & w % n == 0
    if h % n == 0 or w % n == 0:
        raise ValueError("n: %d can not be division by h: %d or w: %d" %(n, h, w))

    images = images.unfold(2, kernel_size, kernel_stride).unfold(3, kernel_size, kernel_stride)
    images = images.contiguous().view(images.size(0), images.size(1), -1, images.size(4), images.size(5))
    images = images.permute((0,2,1,3,4))
    ## shape [n, k, c*h/n*w/n]
    images = images.view(images.size(0), images.size(1), -1)

    edges = edges.unfold(2, kernel_size, kernel_stride).unfold(3, kernel_size, kernel_stride)
    edges = edges.contiguous().view(edges.size(0), edges.size(1), -1, edges.size(4), edges.size(5))
    edges = edges.permute((0,2,1,3,4))
    ## shape [n, c*h/n*w/n, k]
    edges = edges.view(edges.size(0), edges.size(1), -1).permute((0,2,1))

    ## shape [n, k, k]
    relation_matrix = torch.matmul(images, edges)
    return relation_matrix


class RNLLoss(nn.Module):
    def __init__(self, args):
        super(RNLLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, fake, real):
        #fake : tuple (SR, F_Diff)
        #real : tuple (HR, Diff)
        sr, fake_diff = fake
        hr, diff = real

        r1_matrix = pairwiseFunc(sr, fake_diff, args.rloss_n)
        r2_matrix = pairwiseFunc(hr, diff, args.rloss_n)

        loss = self.l1_loss(r1_matrix, r2_matrix)

        return loss
