import os
import glob
import subprocess
from tqdm import tqdm

# INPUT SHOULD BE BOTH T1 and T1N4
# Get the T1s from the text file
with open('qa_all_T1s.txt', 'r') as f:
    T1s = f.read().splitlines()
T1s.sort()

# Register all the T1s
MNI_atlas = '/nfs/masi/newlinnr/cr3/MNI152NLinSym.nii'
for T1 in tqdm(T1s):
    # Output folder for MNI 
    project_folder = T1.split('sub')[0]
    T1file = T1.split('/')[-1]
    sub = T1file.split('_')[0]
    ses = T1file.split('_')[1]

    # Check if sess is present and create the output derivatives folder
    if ses[0] == 's':
        out_folder = project_folder + 'derivatives/T1MNIRegistered/' + sub + '/' + ses + '/anat'
    else:
        out_folder = project_folder + 'derivatives/T1MNIRegistered/' + sub + '/anat'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Output path for N4 bias correction
    out_file = out_folder + '/' + T1file

    # T1 to MNI
    print("Running T12mni" + T1)
    bashCommand = "sh /nfs/masi/kanakap/projects/DeepN4/t12mni.sh " + T1 + " " + MNI_atlas + " "+ out_file
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Written to " + out_file)
