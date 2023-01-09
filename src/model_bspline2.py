import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def bspline_kernel_3d(sigma=[1, 1, 1], order=2, asTensor=False, dtype=torch.float32, device='gpu'):
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1
    for i in range(1, order + 1):
        # change 2d to 3d
        kernel = F.conv3d(kernel, kernel_ones, padding=(padding).tolist())/(sigma[0]*sigma[1]*sigma[2])
    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()

def get_params(downsample, stride):
    _image_size = np.array([128,128,128])
    downscale = downsample
    _stride = stride
    cp_grid = np.ceil(np.divide(_image_size/(1.0*downscale), _stride)).astype(dtype=int)
    inner_image_size = np.multiply(_stride, cp_grid) - (_stride - 1)
    image_size_diff = inner_image_size -_image_size/(1.0*downscale)
    image_size_diff_floor = np.floor( (np.abs(image_size_diff)/2))*np.sign(image_size_diff)
    _crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
    _crop_end = image_size_diff_floor
    return _crop_start,_crop_end

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

    def get_bspline_kernel(self, spacing, order=3):
        '''
        :param order init: bspline order, default to 3
        :param spacing tuple of int: spacing between control points along h and w.
        :return:  kernel matrix
        '''
        self._dtype = torch.float32
        self._device = 'cuda:0'
        self._kernel = bspline_kernel_3d(spacing, order=order, asTensor=True, dtype=self._dtype, device=self._device)
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()
        # self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel.unsqueeze_(0) #pk
        self._kernel = self._kernel.repeat(64,1,1,1) #pk
        self._kernel.unsqueeze_(1) #pk
        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)
        return self._kernel, self._padding

        
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
        s8 = F.relu(self.l8_c2(F.relu(self.l8_c1(c8)))) # [1, 64, 64, 64, 64]

        #get the bspline kernal. spacing between control points along h and w 
        interpolation_kernel, padding = self.get_bspline_kernel(spacing=(2,2,2)) # [64, 1, 5, 5, 5]

        # run the conv w bspline
        output = F.conv_transpose3d(s8, interpolation_kernel,padding=1, stride=1, groups=1) # [1, 1, 66, 66, 66]

        # crop the center
        _crop_start, _crop_end = get_params(downsample=2,stride=1)
        stride=[1,1,1]
        slice1_s = int(stride[0] + _crop_start[0])
        slice1_e = int(-stride[0] - _crop_end[0])
        slice2_s = int(stride[1] + _crop_start[1])
        slice2_e = int(-stride[1] - _crop_end[1])
        slice3_s = int(stride[2] + _crop_start[2])
        slice3_e = int(-stride[2] - _crop_end[2])
        output_cropped = output[:,:, slice1_s:slice1_e , slice2_s:slice2_e , slice3_s:slice3_e ] # [1, 1, 64, 64, 64]

        # scale to input size
        scale_factor_d = 2
        scale_factor_h = 2
        scale_factor_w = 2
        upsampler = torch.nn.Upsample(scale_factor=(scale_factor_d, scale_factor_h, scale_factor_w), mode='trilinear',align_corners=False)
        output_scaled = upsampler(output_cropped)
        #output_tmp = 1,1,128,128,128

        # i9 = self.l9_ct1( s8 )
        # c9 = torch.cat((i9,s1),axis=1)
        # s9 = F.relu(self.l9_c2(F.relu(self.l9_c1(c9))))

        # output = F.sigmoid(self.output(s9))
        return output_scaled