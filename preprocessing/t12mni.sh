# register slant and mask whole brain 
#!/bin/bash
T1=$1
N4=$2
MNI_atlas=$3
OUT_T1=$4
OUT_N4=$5

WORKDIR="$(mktemp -d)"

T12MNI=$WORKDIR/T12MNI

AFFLINE=$WORKDIR/T12MNI0GenericAffine.mat
MNIWRAP=$WORKDIR/T12MNI1Warp.nii.gz
TransT12MNI=$OUT_T1
TransN42MNI=$OUT_N4

/nfs/masi/caily/apps/ants/bin/antsRegistrationSyNQuick.sh -f $MNI_atlas -m $T1 -o $T12MNI -n 60 -t r 

/nfs/masi/caily/apps/ants/bin/antsApplyTransforms -d 3 -i $T1 -r $MNI_atlas -o $TransT12MNI -t $MNIWRAP -t $AFFLINE

/nfs/masi/caily/apps/ants/bin/antsApplyTransforms -d 3 -i $N4 -r $MNI_atlas -o $TransN42MNI -t $MNIWRAP -t $AFFLINE

rm -r $WORKDIR



