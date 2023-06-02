import os
import torch
import argparse as ap
from utils import *
from model_all import *
from nilearn.image import resample_img
from scipy.ndimage import gaussian_filter

def load( subj ):

    input_data = nib.load(subj).get_fdata()
    input_data, [lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ] = pad(input_data, 128)
    in_max = np.percentile(input_data[np.nonzero(input_data)], 99.99)
    input_data = normalize_img(input_data, in_max, 0, 1, 0)
    input_data = np.squeeze(input_data)
    input_vols = np.zeros((1,1, 128, 128, 128))
    input_vols[0,0,:,:,:] = input_data

    return torch.from_numpy(input_vols).float(), lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max

# Inference function
def pred_model( input_path, checkpoint_file ):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

    # Get model checkpoint
    model = Synbo_UNet3D(1, 1).to(device)
    model = load_model(model, checkpoint_file)
    model.eval()
    in_features, lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ, in_max = load(input_path)

    in_features = in_features.to(device)
    logfield = model(in_features, device)
    field = torch.exp(logfield)

    field = field.cpu()
    input_data = in_features.cpu()  

    # Reshape
    field = field.squeeze()
    field_ny = field.detach().numpy()
    input_data = input_data.squeeze()

    org_data = nib.load(input_path).get_fdata()
    final_field = np.zeros([org_data.shape[0], org_data.shape[1], org_data.shape[2]])
    final_field[rx:rX,ry:rY,rz:rZ] = field_ny[lx:lX,ly:lY,lz:lZ]

    final_input = np.zeros([org_data.shape[0], org_data.shape[1], org_data.shape[2]])
    final_input[rx:rX,ry:rY,rz:rZ] = input_data[lx:lX,ly:lY,lz:lZ]

    # Compute corrected image
    final_field = gaussian_filter(final_field, sigma=3)
    final_corrected = final_input / final_field
    final_corrected = unnormalize_img(final_corrected, in_max, 0, 1, 0)

    return final_corrected, final_field
    
def main():

    # Resample data
    parser = ap.ArgumentParser(description='DeepN4: Deep learning based N4 correction')
    parser.add_argument('in_file', help='Input T1 image filename')
    parser.add_argument('out_file', help='Output filename')
    parser.add_argument('--bias_file', metavar='string', default='off', help='Bias field filename')
    args = parser.parse_args()

    input_file = args.in_dir
    output_file = args.out_dir
    bias_file = args.bias_file

    print('INPUT FILE: {}'.format(input_file))
    print('OUTPUT FILE: {}'.format(output_file))

    resample_file = '/tmp/resampled.nii.gz'
    x_res, y_res, z_res = 2, 2, 2
    os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(input_file, resample_file, x_res, y_res, z_res))

    print('CORRECTING FOR BIAS FIELD')
    final_corrected, final_field = pred_model(resample_file, checkpoint_file='/APPS/checkpoint_epoch_264')

    # Save
    # corrected image 
    ref = nib.load(resample_file)
    nii = nib.Nifti1Image(final_corrected, affine=ref.affine, header=ref.header)
    # nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/data/corrected_IXI015-HH-1258-T1.nii.gz')

    # bias field 
    if bias_file != 'off':
        nii = nib.Nifti1Image(final_field, affine=ref.affine, header=ref.header)
        nib.save(nii, bias_file)

    # Resample back to orginal resolution 
    ref = nib.load(input_file)
    output_img = resample_img(nib.Nifti1Image(final_corrected, nib.load(resample_file).affine), target_affine=ref.affine, target_shape=ref.shape)
    nib.save(output_img, output_file)
    os.remove('/tmp/resampled.nii.gz')
    print('DONE')





    

