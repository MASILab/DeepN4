import torch
import numpy as np
import nibabel as nib
from torch.autograd import Variable
from nilearn.image import resample_img
from scipy.ndimage import gaussian_filter
from dipy.utils.optpkg import optional_package
tf, have_tf, _ = optional_package('tensorflow')


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
input_file  = '/nfs/masi/kanakap/projects/DeepN4/data/IXI015-HH-1258-T1.nii.gz'
checkpoint_file = "/nfs/masi/kanakap/projects/DeepN4/src/trained_model_tf/checkpoint_epoch_264.pd"

# Preprocess input data (resample, normalize, and pad)
new_voxel_size = [2, 2, 2]  
resampled_T1 = resample_img(input_file, target_affine=np.diag(new_voxel_size))
in_features, lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max = load_resample(resampled_T1)

print('hi1')
# Load the model 
model = tf.saved_model.load(checkpoint_file)

print('hi2')
# Run the model to get the bias field
logfield = model(input_1=Variable(in_features))
field = np.exp(logfield['120'])
field = field.squeeze()

print('hi3')
# Postprocess predicted field (reshape - unpad, smooth the field, upsample)
org_data = resampled_T1.get_fdata()
final_field = np.zeros([org_data.shape[0], org_data.shape[1], org_data.shape[2]])
final_field[rx:rX,ry:rY,rz:rZ] = field[lx:lX,ly:lY,lz:lZ]
final_fields = gaussian_filter(final_field, sigma=3)
ref = nib.load(input_file)
upsample_final_field = resample_img(nib.Nifti1Image(final_fields,resampled_T1.affine), target_affine=ref.affine, target_shape=ref.shape)

print('hi4')
# Correct the image
upsample_data = upsample_final_field.get_fdata()
ref_data = ref.get_fdata()
with np.errstate(divide='ignore', invalid='ignore'):
    final_corrected = np.where(upsample_data != 0, ref_data / upsample_data, 0)

# Save the corrected image
ref = nib.load(input_file)
nii = nib.Nifti1Image(final_corrected, affine=ref.affine, header=ref.header)
nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/data/tf5_bfup_corrected_IXI015-HH-1258-T1.nii.gz')

nii = upsample_final_field 
nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/data/tf5_bfup_predicted_field_IXI015-HH-1258-T1.nii.gz')