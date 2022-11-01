PROJECT_PATH=/nfs2/harmonization/BIDS/$1

# To get the count of MPRAGE in a project
ls $PROJECT_PATH/sub*/ses*/anat/*_T1w.nii | wc -l
#ls $PROJECT_PATH/sub*/anat/*_T1w.nii.gz | wc -l

# Exit if failed else write output to a file 
status_code=$?

if [ $? -eq 0 ]
then
echo "Writting T1s"
ls $PROJECT_PATH/sub*/ses*/anat/*_T1w.nii >> /nfs/masi/kanakap/projects/DeepN4/2all_T1s.txt
#ls $PROJECT_PATH/sub*/anat/*_T1w.nii.gz >> /nfs/masi/kanakap/projects/DeepN4/2all_T1s.txt
else
echo "Count failed"
exit 1
fi


