import torch
import numpy as np
import nibabel as nib
from torch import nn
from nilearn.image import resample_img
from scipy.ndimage import gaussian_filter

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

    def forward(self, x, device): #def forward(self, x, device):
        # Encodes
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
        out  = self.dl(d0)
        return out 
    
def load_model(model, path):

    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def pad(img, sz):

    tmp = np.zeros((sz, sz, sz))
    diff = int((sz-img.shape[0])/2)
    lx = max(diff,0)
    lX = min(img.shape[0]+diff,sz)

    diff = (img.shape[0]-sz) / 2
    rx = max(int(np.floor(diff)),0)
    rX = min(img.shape[0]-int(np.ceil(diff)),img.shape[0])

    diff = int((sz - img.shape[1]) / 2)
    ly = max(diff, 0)
    lY = min(img.shape[1] + diff, sz)

    diff = (img.shape[1] - sz) / 2
    ry = max(int(np.floor(diff)), 0)
    rY = min(img.shape[1] - int(np.ceil(diff)), img.shape[1])

    diff = int((sz - img.shape[2]) / 2)
    lz = max(diff, 0)
    lZ = min(img.shape[2] + diff, sz)

    diff = (img.shape[2] - sz) / 2
    rz = max(int(np.floor(diff)), 0)
    rZ = min(img.shape[2] - int(np.ceil(diff)), img.shape[2])

    tmp[lx:lX,ly:lY,lz:lZ] = img[rx:rX,ry:rY,rz:rZ]

    return tmp, [lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ]

def normalize_img(img, max_img, min_img, a_max, a_min):

    img = (img - min_img)/(max_img - min_img)
    img = np.clip(img, a_max=a_max, a_min=a_min)

    return img

def load_resample( subj ):

    input_data, [lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ] = pad(subj.get_fdata(), 128)
    in_max = np.percentile(input_data[np.nonzero(input_data)], 99.99)
    input_data = normalize_img(input_data, in_max, 0, 1, 0)
    input_data = np.squeeze(input_data)
    input_vols = np.zeros((1,1, 128, 128, 128))
    input_vols[0,0,:,:,:] = input_data

    return torch.from_numpy(input_vols).float(), lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max

# Input data
input_file = '/nfs/masi/kanakap/projects/DeepN4/data/IXI015-HH-1258-T1.nii.gz'
checkpoint_file='/nfs/masi/kanakap/projects/DeepN4/src/trained_model_Synbo_UNet3D/checkpoint_epoch_264'

# Preprocess input data (resample, normalize, and pad)
new_voxel_size = [2, 2, 2]  
resampled_T1 = resample_img(input_file, target_affine=np.diag(new_voxel_size))
in_features, lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max = load_resample(resampled_T1)

# Set up CUDA if available, seed for reproducibility, and choose device
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

# Load the model 
model = Synbo_UNet3D(1, 1).to(device)
model = load_model(model, checkpoint_file)
model.eval()

# Run the model to get the bias field
in_features = in_features.to(device)
logfield = model(in_features, device)
field = torch.exp(logfield)
field = field.cpu()
field = field.squeeze()
field_np = field.detach().numpy()

# Postprocess predicted field (reshape - unpad, smooth the field, upsample)
org_data = resampled_T1.get_fdata()
final_field = np.zeros([org_data.shape[0], org_data.shape[1], org_data.shape[2]])
final_field[rx:rX,ry:rY,rz:rZ] = field_np[lx:lX,ly:lY,lz:lZ]
final_field = gaussian_filter(final_field, sigma=3)
ref = nib.load(input_file)
upsample_final_field = resample_img(nib.Nifti1Image(final_field,resampled_T1.affine), target_affine=ref.affine, target_shape=ref.shape)

# Correct the image 
upsample_data = upsample_final_field.get_fdata()
ref_data = ref.get_fdata()
with np.errstate(divide='ignore', invalid='ignore'):
    final_corrected = np.where(upsample_data != 0, ref_data / upsample_data, 0)

# Save the corrected image
ref = nib.load(input_file)
nii = nib.Nifti1Image(final_corrected, affine=ref.affine, header=ref.header)
nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/data/new6_bfup_corrected_IXI015-HH-1258-T1.nii.gz')

nii = upsample_final_field
nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/data/new6_bfup_predicted_field_IXI015-HH-1258-T1.nii.gz')