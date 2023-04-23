import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from scratch.bspline_fix import *
from scratch.pad_within import *

class Synbo_UNet3D(nn.Module):
    def __init__(self, n_in, n_out):
        super(Synbo_UNet3D, self).__init__()
        # Encoder
        c = 32
        self.ec0 = self.encoder_block(      n_in,    1*c, kernel_size=3, stride=1, padding=1)
        self.ec1 = self.encoder_block(        c,    c*2, kernel_size=3, stride=1, padding=1)
        self.pool0 = nn.MaxPool3d(2)
        self.ec2 = self.encoder_block(        c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.ec3 = self.encoder_block(        c*2,   c*4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.ec4 = self.encoder_block(       c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.ec5 = self.encoder_block(       c*4,   c*8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.ec6 = self.encoder_block(       c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.ec7 = self.encoder_block(       c*8,   c*16, kernel_size=3, stride=1, padding=1)
        self.el  =          nn.Conv3d(       c*16,   c*16, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.dc9 = self.decoder_block(       c*16,   c*16, kernel_size=2, stride=2, padding=0)
        self.dc8 = self.decoder_block( c*16 + c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.decoder_block(       c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = self.decoder_block(       c*8,   c*8, kernel_size=2, stride=2, padding=0)
        self.dc5 = self.decoder_block( c*8 + c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.decoder_block(       c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = self.decoder_block(       c*4,   c*4, kernel_size=2, stride=2, padding=0)
        self.dc2 = self.decoder_block(  c*4 + c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.decoder_block(        c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.dc0 = self.decoder_block(        c*2, n_out, kernel_size=1, stride=1, padding=0)
        self.dl  = nn.ConvTranspose3d(     n_out, n_out, kernel_size=1, stride=1, padding=0)
        # self.act  = nn.LazyLinear(n_out)
        # self.act = nn.Linear(n_out, n_out)
        # self.dl  = BSplineLayer(     4, 4, n_bases=6, shared_weights=True,bias=False, weighted_sum=False)#, kernel_size=1, stride=1, padding=0)

    def encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def forward(self, x, device):
        # Encode
        e0   = self.ec0(x)
        syn0 = self.ec1(e0)
        del e0

        e1   = self.pool0(syn0)
        e2   = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e1, e2

        e3   = self.pool1(syn1)
        e4   = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5   = self.pool2(syn2)
        e6   = self.ec6(e5)
        e7   = self.ec7(e6)

        # Last layer without relu
        el   = self.el(e7)
        del e5, e6, e7

        # Decode
        d9   = torch.cat((self.dc9(el), syn2), 1)
        del el, syn2

        d8   = self.dc8(d9)
        d7   = self.dc7(d8)
        del d9, d8

        d6   = torch.cat((self.dc6(d7), syn1), 1)
        del d7, syn1

        d5   = self.dc5(d6)
        d4   = self.dc4(d5)
        del d6, d5

        d3   = torch.cat((self.dc3(d4), syn0), 1)
        del d4, syn0

        d2   = self.dc2(d3)
        d1   = self.dc1(d2)
        del d3, d2

        d0   = self.dc0(d1) 
        del d1

        # Last layer without relu
        out  = self.dl(d0)#. reshape(-1, 1*128*128*128)
        return out #self.act(out)

class Deep_Synbo_UNet3D(nn.Module):
    def __init__(self, n_in, n_out):
        super(Deep_Synbo_UNet3D, self).__init__()
        # Encoder
        c = 32
        self.ec0 = self.encoder_block(      n_in,    1*c, kernel_size=3, stride=1, padding=1)
        self.ec1 = self.encoder_block(        c,    c*2, kernel_size=3, stride=1, padding=1)
        self.pool0 = nn.MaxPool3d(2)
        self.ec2 = self.encoder_block(        c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.ec3 = self.encoder_block(        c*2,   c*4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.ec4 = self.encoder_block(       c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.ec5 = self.encoder_block(       c*4,   c*8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.ec6 = self.encoder_block(       c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.ec7 = self.encoder_block(       c*8,   c*16, kernel_size=3, stride=1, padding=1)
        
        self.pool3 = nn.MaxPool3d(2)
        self.ec8 = self.encoder_block(       c*16,   c*16, kernel_size=3, stride=1, padding=1)
        self.ec9 = self.encoder_block(       c*16,   c*32, kernel_size=3, stride=1, padding=1)
        
        self.el  =          nn.Conv3d(       c*32,   c*32, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.dc12 = self.decoder_block(       c*32,   c*32, kernel_size=2, stride=2, padding=0)
        self.dc11 = self.decoder_block( c*32 + c*16,   c*16, kernel_size=3, stride=1, padding=1)
        self.dc10 = self.decoder_block(       c*16,   c*16, kernel_size=3, stride=1, padding=1)

        self.dc9 = self.decoder_block(       c*16,   c*16, kernel_size=2, stride=2, padding=0)
        self.dc8 = self.decoder_block( c*16 + c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.decoder_block(       c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = self.decoder_block(       c*8,   c*8, kernel_size=2, stride=2, padding=0)
        self.dc5 = self.decoder_block( c*8 + c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.decoder_block(       c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = self.decoder_block(       c*4,   c*4, kernel_size=2, stride=2, padding=0)
        self.dc2 = self.decoder_block(  c*4 + c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.decoder_block(        c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.dc0 = self.decoder_block(        c*2, n_out, kernel_size=1, stride=1, padding=0)
        self.dl  = nn.ConvTranspose3d(     n_out, n_out, kernel_size=1, stride=1, padding=0)
        # self.act  = nn.LazyLinear(n_out)
        # self.act = nn.Linear(n_out, n_out)
        # self.dl  = BSplineLayer(     4, 4, n_bases=6, shared_weights=True,bias=False, weighted_sum=False)#, kernel_size=1, stride=1, padding=0)

    def encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def forward(self, x, device):
        # Encode
        e0   = self.ec0(x)
        syn0 = self.ec1(e0)
        del e0

        e1   = self.pool0(syn0)
        e2   = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e1, e2

        e3   = self.pool1(syn1)
        e4   = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5   = self.pool2(syn2)
        e6   = self.ec6(e5)
        syn3   = self.ec7(e6)
        del e5, e6

        e7   = self.pool3(syn3)
        e8   = self.ec8(e7)
        e9   = self.ec9(e8)

        # Last layer without relu
        el   = self.el(e9)
        del e7, e8, e9

        # Decode
        d12   = torch.cat((self.dc12(el), syn3), 1)
        del el, syn3

        d11   = self.dc11(d12)
        d10   = self.dc10(d11)
        del d12, d11

        d9   = torch.cat((self.dc9(d10), syn2), 1)
        del d10, syn2

        d8   = self.dc8(d9)
        d7   = self.dc7(d8)
        del d9, d8

        d6   = torch.cat((self.dc6(d7), syn1), 1)
        del d7, syn1

        d5   = self.dc5(d6)
        d4   = self.dc4(d5)
        del d6, d5

        d3   = torch.cat((self.dc3(d4), syn0), 1)
        del d4, syn0

        d2   = self.dc2(d3)
        d1   = self.dc1(d2)
        del d3, d2

        d0   = self.dc0(d1) 
        del d1

        # Last layer without relu
        out  = self.dl(d0)#. reshape(-1, 1*128*128*128)
        return out #self.act(out)

class trad_UNet3D(nn.Module):

    def __init__(self, in_chans, output_chans):
        super(trad_UNet3D, self).__init__()

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
        # self.lastlayer  = nn.ConvTranspose3d(output_chans, output_chans, kernel_size=1, stride=1, padding=0)

    def forward(self,x, device):
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

        # output = F.eLU(self.output(s9))
        out = self.output(s9)
        # final_out = self.lastlayer(out)
        return out

class bspline_UNet3D(nn.Module):

    def __init__(self, in_chans, output_chans):
        super(bspline_UNet3D, self).__init__()

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

        
    def forward(self, x, device):
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
        s8 = F.relu(self.l8_c2(F.relu(self.l8_c1(c8)))) # [1, 64, 64, 64, 64]

        i9 = self.l9_ct1( s8 )
        c9 = torch.cat((i9,s1),axis=1)
        s9 = F.relu(self.l9_c2(F.relu(self.l9_c1(c9))))

        # img = F.sigmoid(self.output(s9))

        # bspline
        K = bspline_kernel_3d_redo(spacing_between_knots=[2,2,2])
        pad_op = PadWithin3D(stride=2)
        padded_img = pad_op(s8,device)
        K = torch.Tensor(K).to(device)
        K = K.repeat(1,64,1,1,1)
        bspline_output = F.conv3d(padded_img, K, padding="same", stride=1, groups=1)

        return bspline_output

class bspline_Synbo_UNet3D(nn.Module):
    def __init__(self, n_in, n_out):
        super(Synbo_UNet3D, self).__init__()
        # Encoder
        c = 32
        self.ec0 = self.encoder_block(      n_in,    1*c, kernel_size=3, stride=1, padding=1)
        self.ec1 = self.encoder_block(        c,    c*2, kernel_size=3, stride=1, padding=1)
        self.pool0 = nn.MaxPool3d(2)
        self.ec2 = self.encoder_block(        c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.ec3 = self.encoder_block(        c*2,   c*4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.ec4 = self.encoder_block(       c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.ec5 = self.encoder_block(       c*4,   c*8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.ec6 = self.encoder_block(       c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.ec7 = self.encoder_block(       c*8,   c*16, kernel_size=3, stride=1, padding=1)
        self.el  =          nn.Conv3d(       c*16,   c*16, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.dc9 = self.decoder_block(       c*16,   c*16, kernel_size=2, stride=2, padding=0)
        self.dc8 = self.decoder_block( c*16 + c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.decoder_block(       c*8,   c*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = self.decoder_block(       c*8,   c*8, kernel_size=2, stride=2, padding=0)
        self.dc5 = self.decoder_block( c*8 + c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.decoder_block(       c*4,   c*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = self.decoder_block(       c*4,   c*4, kernel_size=2, stride=2, padding=0)
        self.dc2 = self.decoder_block(  c*4 + c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.decoder_block(        c*2,    c*2, kernel_size=3, stride=1, padding=1)
        self.dc0 = self.decoder_block(        c*2, n_out, kernel_size=1, stride=1, padding=0)
        self.dl  = nn.ConvTranspose3d(     n_out, n_out, kernel_size=1, stride=1, padding=0)
        # self.act  = nn.LazyLinear(n_out)
        # self.act = nn.Linear(n_out, n_out)
        # self.dl  = BSplineLayer(     4, 4, n_bases=6, shared_weights=True,bias=False, weighted_sum=False)#, kernel_size=1, stride=1, padding=0)

    def encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU())
        return layer

    def forward(self, x, device):
        # Encode
        e0   = self.ec0(x)
        syn0 = self.ec1(e0)
        del e0

        e1   = self.pool0(syn0)
        e2   = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e1, e2

        e3   = self.pool1(syn1)
        e4   = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5   = self.pool2(syn2)
        e6   = self.ec6(e5)
        e7   = self.ec7(e6)

        # Last layer without relu
        el   = self.el(e7)
        del e5, e6, e7

        # Decode
        d9   = torch.cat((self.dc9(el), syn2), 1)
        del el, syn2

        d8   = self.dc8(d9)
        d7   = self.dc7(d8)
        del d9, d8

        d6   = torch.cat((self.dc6(d7), syn1), 1)
        del d7, syn1

        d5   = self.dc5(d6)
        d4   = self.dc4(d5)
        del d6, d5

        d3   = torch.cat((self.dc3(d4), syn0), 1)
        del d4, syn0

        d2   = self.dc2(d3)
        d1   = self.dc1(d2)
        del d3, d2

        d0   = self.dc0(d1) 
        del d1

        # Last layer without relu
        out  = self.dl(d0)#. reshape(-1, 1*128*128*128)

        # bspline
        K = bspline_kernel_3d_redo(spacing_between_knots=[2,2,2])
        pad_op = PadWithin3D(stride=2)
        padded_img = pad_op(d1,device)
        K = torch.Tensor(K).to(device)
        K = K.repeat(1,64,1,1,1)
        bspline_output = F.conv3d(padded_img, K, padding="same", stride=1, groups=1)

        return bspline_output
