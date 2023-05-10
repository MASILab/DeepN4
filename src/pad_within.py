import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class PadWithin2D(nn.Module):
    # takes [B, C, H, W]
    # outputs [B, C, H*stride, W*stride], backfills `stride`-many zeros between inputs
    def __init__(self, stride=2):
        super(PadWithin2D, self).__init__()
        self.stride = stride
        self.w = torch.zeros(self.stride, self.stride)
        self.w[0,0] = 1

    def forward(self, feats):
        return F.conv_transpose2d(
            feats, self.w.expand(feats.shape[1], 1, self.stride, self.stride), stride=self.stride, groups=feats.shape[1]
        )

class PadWithin3D(nn.Module):
    # takes [B, C, H, W, D]
    # outputs [B ,C, H*stride, W*stride, D*stride], backfills `stride`-many zeros between inputs
    def __init__(self, stride=2):
        super(PadWithin3D, self).__init__()
        self.stride = stride
        self.w = torch.zeros(self.stride, self.stride, self.stride)
        self.w[0,0,0] = 1

    def forward(self, feats, device):
        return F.conv_transpose3d(
            feats, self.w.expand(feats.shape[1], 1, self.stride, self.stride, self.stride).to(device), stride=self.stride, groups=feats.shape[1]
        )
