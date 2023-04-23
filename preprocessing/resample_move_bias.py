import os
import glob 
import nibabel as nib
import time
import sys
from tqdm import tqdm

root_src = "/nfs2/harmonization/BIDS/"+sys.argv[1]+"/derivatives"
Bs = []
for files in glob.glob(root_src +'/*/T1_BiasField/*.nii.gz'):
   Bs.append(files)
Bs.sort()

root = "/nfs2/harmonization/BIDS/"+sys.argv[1]
T1s = []
for files in glob.glob(root +'/*/anat/*_T1w.nii.gz'):
   T1s.append(files)
T1s.sort()

#Target
N4_T1s = []
for files in glob.glob(root_src +'/*/T1_N4Corrected/*.nii.gz'):
   N4_T1s.append(files)
N4_T1s.sort()

# SLANT 
root_src = "/nfs2/harmonization/BIDS/"+sys.argv[1]+"/derivatives"
T1s = []
for files in glob.glob(root_src +'/*/*/SLANT-run2/FinalResult/*.nii.gz'):
    T1s.append(files)
T1s.sort()


x_res = 2
y_res = 2
z_res = 2



for i in tqdm(range(len(T1s))):

    img_b = Bs[i]
    filenameb = Bs[i].split('/')[-1]
    output_fileb = '/home/local/VANDERBILT/kanakap/deepN4_data/bias_more/' + filenameb
    os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(img_b, output_fileb, x_res, y_res, z_res))
    print('Resampled '+ str(i) + img_b)
    print('-----------------------------')

    img_MRI = T1s[i]
    filename = T1s[i].split('/')[-1]
    output_file = '/home/local/VANDERBILT/kanakap/deepN4_data/inputs_more/' + filename
    if not os.path.isfile(output_file):
       os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(img_MRI, output_file, x_res, y_res, z_res))

    
    img_N4 = N4_T1s[i]
    filenameN4 = N4_T1s[i].split('/')[-1]
    output_fileN4 = '/home/local/VANDERBILT/kanakap/deepN4_data/labels_more/' + filenameN4
    os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(img_N4, output_fileN4, x_res, y_res, z_res))
    print('Resampled ' + img_MRI +' '+ img_N4)

    # # SLANT
    img_MRI = T1s[i]
    sub = T1s[i].split('/')[6]
    sess = T1s[i].split('/')[7]
    filename = sub + '_' + sess + '_run-02' + '_T1w.nii.gz' 
    # filename = sub + '_T1w.nii.gz' # hcp
    print(filename)
    output_file = '/home/local/VANDERBILT/kanakap/deepN4_data/mask_more/' + filename
    if not os.path.isfile(output_file):
        os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(img_MRI, output_file, x_res, y_res, z_res))