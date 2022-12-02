import os
import glob 
import nibabel as nib
import time

# Input image
#root_src = "/nfs2/harmonization/BIDS/ADNI_tmp"
root_src = "/nfs2/harmonization/BIDS/ADNI_tmp/derivatives/T1MNIRegistered"
T1s = []
for files in glob.glob(root_src +'/*/*/anat/*.nii.gz'):
    T1s.append(files)
T1s.sort()

# Target
#root_out = "/nfs2/harmonization/BIDS/ADNI_tmp/derivatives/N4BiasFieldCorrection"
root_out = "/nfs2/harmonization/BIDS/ADNI_tmp/derivatives/T1N4_MNIRegistered"
N4_T1s = []
for files in glob.glob(root_out +'/*/*/anat/*.nii.gz'):
    N4_T1s.append(files)
N4_T1s.sort()

x_res = 2
y_res = 2
z_res = 2



for i in range(len(N4_T1s)):

    img_MRI = T1s[i]
    filename = T1s[i].split('/')[-1]
    output_file = '/home/local/VANDERBILT/kanakap/deepN4_data_MNI/inputs/' + filename
    os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(img_MRI, output_file, x_res, y_res, z_res))

    
    img_N4 = N4_T1s[i]
    filenameN4 = N4_T1s[i].split('/')[-1]
    output_fileN4 = '/home/local/VANDERBILT/kanakap/deepN4_data_MNI/labels/' + filenameN4
    os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(img_N4, output_fileN4, x_res, y_res, z_res))
    print('Resampled ' + img_MRI +' '+ img_N4)
