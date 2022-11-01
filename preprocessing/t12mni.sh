# register slant and mask whole brain 
#!/bin/bash
T1=$1
MNI_atlas=$2
OUT_T1=$3

WORKDIR="$(mktemp -d)"

T12MNI=$WORKDIR/T12MNI

AFFLINE=$WORKDIR/T12MNI0GenericAffine.mat
MNIWRAP=$WORKDIR/T12MNI1Warp.nii.gz
TransT12MNI=$OUT_T1

/nfs/masi/caily/apps/ants/bin/antsRegistrationSyNQuick.sh -f $MNI_atlas -m $T1 -o $T12MNI -n 4

/nfs/masi/caily/apps/ants/bin/antsApplyTransforms -d 3 -i $T1 -r $MNI_atlas -o $TransT12MNI -t $MNIWRAP -t $AFFLINE

rm -r $WORKDIR



