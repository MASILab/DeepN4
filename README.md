# DeepN4
Pytorch implementation of DeepN4, from the paper: DeepN4: Learning ITKN4 Bias Field Correction for T1 weighted Images 
We propose **DeepN4**, a 3D UNet to generate Bias Field for T1w images and inturn correct T1w images for inhomogeneities.

<p align="center">
<img src="Figures/fig.png" width=100% height=40% 
class="center">
</p>

## Prequisite installation
* **ADD**:  Please look into the ### for creating conda environment and package installation procedures. 

## Training
1. TODO dataset file 
2. Run 
```
python test_seg.py --root path_to_image_folder --output path_to_output \
--dataset flare --network 3DUXNET --trained_weights path_to_trained_weights \
```

## Evaluation
Efficient evaulation can be performed for the public datasets as follows:

---






