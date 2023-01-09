import torch
import torch.nn as nn


class crblock(nn.Module):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(64*inc, outc*64, kernel_size=3, stride=1, padding=0, dilation=2)
        self.bn1 = nn.InstanceNorm3d(outc)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv3d(64*outc, inc*64, kernel_size=3, stride=1, padding=4, dilation=2)
        self.bn2 = nn.InstanceNorm3d(inc)

    def forward(self, input):
        x = self.act(self.bn1(self.conv1(input)))
        # import pdb; pdb.set_trace()
        x = self.act(self.bn2(self.conv2(x)))
        return x + input

class BiasNet(nn.Module):
    def __init__(self,inc, outc) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(inc, 64*outc, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv3d(inc, 64*outc, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv3d(inc, 64*outc, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv3d(inc, 64*outc, kernel_size=3, stride=1, padding=0)

        self.crblocks = nn.Sequential(*[crblock(inc, outc) for i in range(15)])
        self.conv5 = nn.Conv3d(64*outc, inc, kernel_size=1, padding=1, stride=1)

    def forward(self,input1,input2,input3,input4):


        i1 = self.conv1(input1)
        del input1
        i2 = self.conv2(input2)
        del input2
        i3 = self.conv3(input3)
        del input3
        # i4 = self.conv4(input4)
        l1 = i1 + i2# + i3 #+ i4
        del i1, i2, i3
        l2 = self.crblocks(l1)
        del l1
        return self.conv5(l2)


# convblk = lambda inc, outc: nn.Sequential(
#     nn.Conv2d(inc, outc, 5, padding=2),
#     nn.BatchNorm2d(outc),
#     nn.LeakyReLU(),
# )

# class BiasNet(nn.Module):
    
#     def __init__(self, n_channel: int):
#         super(BiasNet, self).__init__()
#         self.conv = nn.Sequential(
#             convblk(n_channel, 32),
#             convblk(32, 64),
#             convblk(64, 128),
#             convblk(128, 256),
#             convblk(256, 128),
#             convblk(128, 64),
#             convblk(128, 64),
#             convblk(64, 32),
#             nn.Conv2d(32,n_channel, 3, padding =1)
#         )

#     def forward(self, x, t):
#         return self.conv(x)