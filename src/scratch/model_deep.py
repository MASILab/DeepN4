import torch
from torch import nn
from torch.nn import functional as F


class UNet_Eyeballs_Sandbox(nn.Module):

    def __init__(self, in_chans, output_chans):
        super(UNet_Eyeballs_Sandbox, self).__init__()

        #to match previous notation, it's 1 indexed...
        #sorry
        self.l1_c1 = nn.Conv3d(in_chans, 32, kernel_size=3, stride=1, padding=1)
        self.l1_c2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.l1_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.l2_c1 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.l2_c2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l2_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.l3_c1 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l3_c2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l3_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.l4_c1 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l4_c2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l4_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        #center
        self.l5_c1 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l5_c2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)

        #upsampling sequences
        self.l6_ct1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0)
        #cats here
        self.l6_c1 = nn.Conv3d(256+256, 256, kernel_size=3, stride=1, padding=1)
        self.l6_c2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.l7_ct1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
        self.l7_c1 = nn.Conv3d(128+128, 128, kernel_size=3, stride=1, padding=1)
        self.l7_c2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.l8_ct1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.l8_c1 = nn.Conv3d(64+64, 64, kernel_size=3, stride=1, padding=1)
        self.l8_c2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        self.l9_ct1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.l9_c1 = nn.Conv3d(32+32, 32, kernel_size=3, stride=1, padding=1)
        self.l9_c2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        self.output = nn.Conv3d(32, output_chans, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        s1 = F.relu(self.l1_c2(F.relu(self.l1_c1(x))))
        s2 = F.relu(self.l2_c2(F.relu(self.l2_c1(self.l1_p1(s1)))))
        s3 = F.relu(self.l3_c2(F.relu(self.l3_c1(self.l2_p1(s2)))))
        s4 = F.relu(self.l4_c2(F.relu(self.l4_c1(self.l3_p1(s3)))))
    
        s5 = F.relu(self.l5_c2(F.relu(self.l5_c1(self.l4_p1(s4)))))

        i6 = self.l6_ct1( s5 )
        c6 = torch.cat((i6,s4),axis=1)
        s6 = F.relu(self.l6_c2(F.relu(self.l6_c1(c6))))
        
        i7 = self.l7_ct1( s6 )
        c7 = torch.cat((i7,s3),axis=1)
        s7 = F.relu(self.l7_c2(F.relu(self.l7_c1(c7))))

        i8 = self.l8_ct1( s7 )
        c8 = torch.cat((i8,s2),axis=1)
        s8 = F.relu(self.l8_c2(F.relu(self.l8_c1(c8))))

        i9 = self.l9_ct1( s8 )
        c9 = torch.cat((i9,s1),axis=1)
        s9 = F.relu(self.l9_c2(F.relu(self.l9_c1(c9))))

        output = F.sigmoid(self.output(s9))
        return output

class UNet_Eyeballs_Scalar_append_at_start(nn.Module):

    def __init__(self, in_chans, scalar_chans, output_chans):
        super(UNet_Eyeballs_Scalar_append_at_start, self).__init__()

        self.scalar_chans = scalar_chans
        #to match previous notation, it's 1 indexed...
        #sorry
        self.l1_c1 = nn.Conv3d(in_chans+scalar_chans, 32, kernel_size=3, stride=1, padding=1)
        self.l1_c2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.l1_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.l2_c1 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.l2_c2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l2_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.l3_c1 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l3_c2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l3_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.l4_c1 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l4_c2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l4_p1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        #center
        self.l5_c1 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l5_c2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)

        #upsampling sequences
        self.l6_ct1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0)
        #cats here
        self.l6_c1 = nn.Conv3d(256+256, 256, kernel_size=3, stride=1, padding=1)
        self.l6_c2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.l7_ct1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0)
        self.l7_c1 = nn.Conv3d(128+128, 128, kernel_size=3, stride=1, padding=1)
        self.l7_c2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.l8_ct1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0)
        self.l8_c1 = nn.Conv3d(64+64, 64, kernel_size=3, stride=1, padding=1)
        self.l8_c2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        self.l9_ct1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0)
        self.l9_c1 = nn.Conv3d(32+32, 32, kernel_size=3, stride=1, padding=1)
        self.l9_c2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        self.output = nn.Conv3d(32, output_chans, kernel_size=1, stride=1, padding=0)

    def forward(self,x,s):
        #s is the scalar

        #s must be single channel
        #TODO: generalize to vectors?
        #TODO: maybe make this a function?
        # add on spatial dims
        s = s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shape = [x.shape[0],self.scalar_chans,x.shape[2],x.shape[3],x.shape[4]]
        x_s = torch.cat([x,s*torch.ones(shape,device=s.device)], axis=1) # concat to axis 1
        s1 = F.relu(self.l1_c2(F.relu(self.l1_c1(x_s))))
        s2 = F.relu(self.l2_c2(F.relu(self.l2_c1(self.l1_p1(s1)))))
        s3 = F.relu(self.l3_c2(F.relu(self.l3_c1(self.l2_p1(s2)))))
        s4 = F.relu(self.l4_c2(F.relu(self.l4_c1(self.l3_p1(s3)))))

        s5 = F.relu(self.l5_c2(F.relu(self.l5_c1(self.l4_p1(s4)))))

        i6 = self.l6_ct1( s5 )
        c6 = torch.cat((i6,s4),axis=1)
        s6 = F.relu(self.l6_c2(F.relu(self.l6_c1(c6))))
        
        i7 = self.l7_ct1( s6 )
        c7 = torch.cat((i7,s3),axis=1)
        s7 = F.relu(self.l7_c2(F.relu(self.l7_c1(c7))))

        i8 = self.l8_ct1( s7 )
        c8 = torch.cat((i8,s2),axis=1)
        s8 = F.relu(self.l8_c2(F.relu(self.l8_c1(c8))))

        i9 = self.l9_ct1( s8 )
        c9 = torch.cat((i9,s1),axis=1)
        s9 = F.relu(self.l9_c2(F.relu(self.l9_c1(c9))))

        output = F.sigmoid(self.output(s9))
        return output

