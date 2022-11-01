import os
import glob
import subprocess
from tqdm import tqdm



# Get the T1s in BIDS folders
# root_src = "/nfs2/harmonization/BIDS/ADNI_tmp"
# root_out = "/nfs2/harmonization/BIDS/ADNI_tmp/derivatives/N4BiasFieldCorrection"
# project = 'ADNI_tmp'
# T1s = []
# for files in glob.glob(root_src +'/*/*/anat/*.nii.gz'):
#     T1s.append(files)

# Get the T1s from the text file 
with open('noad_qa_all_T1s.txt', 'r') as f:
    T1s = f.read().splitlines()
T1s.sort()

# For all T1s run N4 bias correction
for T1 in tqdm(T1s):
    # Output folder for N4 bias correction
    project_folder = T1.split('sub')[0]
    T1file = T1.split('/')[-1]
    sub = T1file.split('_')[0]
    ses = T1file.split('_')[1]

    # Check if sess is present and create the output derivatives folder
    if ses[0] == 's':
        out_folder = project_folder + 'derivatives/N4BiasFieldCorrection/' + sub + '/' + ses + '/anat'
    else:
        out_folder = project_folder + 'derivatives/N4BiasFieldCorrection/' + sub + '/anat'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Output path for N4 bias correction
    out_file = out_folder + '/' + T1file
    
    # N4 bias correction
    print("Running N4BiasFieldCorrection" + T1)
    bashCommand = "N4BiasFieldCorrection -d 3 -i " + T1 + " -o " + out_file
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Written to " + out_file)
