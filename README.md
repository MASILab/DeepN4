### [DeepN4: Learning ITKN4 Bias Field Correction for T1 weighted Images]

Official Pytorch implementation of DeepN4
We propose **DeepN4**, a 3D UNet to generate Bias Field for T1w images.
---

<p align="center">
<img src="Figures/Figure_1.png" width=100% height=40% 
class="center">
</p>


 ## Installation
 Please look into the [INSTALL.md](INSTALL.md) for creating conda environment and package installation procedures.

<!-- ✅ ⬜️  -->
## Training
Training and fine-tuning instructions are in [TRAINING.md](TRAINING.md). Pretrained model weights will be uploaded for public usage later on.

<!-- ✅ ⬜️  -->
## Evaluation
Efficient evaulation can be performed for the public datasets as follows:
```
python test_seg.py --root path_to_image_folder --output path_to_output \
--dataset flare --network 3DUXNET --trained_weights path_to_trained_weights \
--mode test --sw_batch_size 4 --overlap 0.7 --gpu 0 --cache_rate 0.2 \
```



