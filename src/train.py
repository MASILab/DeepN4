import os
import torch
import glob
import util
import dataloader
import numpy as np
import torch.nn.functional as F


def train_model(derivatives_path, model, device, optimizer):
    # Train mode
    
    model.train()

    # # Zero gradient
    optimizer.zero_grad()

    # Compute loss
    loss = compute_loss(derivatives_path, model, device)

    # # Compute gradient
    # loss.backward()

    # # Step optimizer
    # optimizer.step()

    # # Return loss
    # return loss.item()


def compute_loss(derivatives_path, model, device):
    # Get predicted images and masks
    img_models = []
    img_targets = []
    # img_masks = []
    #for synb0prep_dir_path in synb0prep_dir_paths:
        # Get data, target, and mask
    #img_data, img_target, img_mask = dataloader.get_data_and_target(synb0prep_dir_path, device)
    img_data, img_target = get_data_and_target(derivatives_path, device)

    # # Pass through model
    # img_model = model(img_data)

    # # Append
    # img_models.append(img_model)
    # img_targets.append(img_target)
    # # img_masks.append(img_mask)

    # # Compute loss
    # loss = torch.zeros(1, 1, device=device) # Initialize to zero

    # # First, get "truth loss"
    # #for idx in range(len(synb0prep_dir_paths)):
    # for idx in range(len(derivatives_path)):
    #     # Get model, target, and mark
    #     img_model = img_models[idx]
    #     img_target = img_targets[idx]
    #     # img_mask = img_masks[idx]

    #     # Compute loss
    #     loss += F.mse_loss(img_model, img_target)

    # # Divide loss by number of synb0prep directories
    # #loss /= len(synb0prep_dir_paths)
    # loss /= len(derivatives_path)

    # # Next, get "difference loss"
    # #if len(synb0prep_dir_paths) == 2:
    # if len(derivatives_path) == 2:
    #     # Get model, target, and mark
    #     img_model1 = img_models[0]
    #     img_model2 = img_models[1]
    #     # img_mask = img_masks[0] & img_masks[1]

    #     # Add difference loss
    #     loss += F.mse_loss(img_model1, img_model2)
    # elif len(synb0prep_dir_paths) == 1:
    # elif len(derivatives_path) == 1:
    #     pass # Don't add any difference loss
    # else:
    #     raise RunTimeError(train_dir_path + ': Only single and double blips are supported')

    # return loss




def validate_model(derivatives_path, model, device):
    # Eval mode
    model.eval()

    # Compute loss
    loss = compute_loss(derivatives_path, model, device)

    # Return loss
    return loss.item()




def get_data_and_target(training_path, device):
    # Get paths
    T1_path = training_path[0] #os.path.join(synb0prep_dir_path, 'T1_norm_lin_atlas_2_5.nii.gz')
    # b0_d_path = os.path.join(synb0prep_dir_path, 'b0_d_lin_atlas_2_5.nii.gz')
    n4_path = training_path[1] #os.path.join(synb0prep_dir_path, 'b0_u_lin_atlas_2_5.nii.gz')
    # mask_path = os.path.join(synb0prep_dir_path, 'mask_lin.nii.gz') 

    # Get image
    img_T1 = np.expand_dims(util.get_nii_img(T1_path), axis=3)
    # # img_b0_d = np.expand_dims(util.get_nii_img(b0_d_path), axis=3)
    img_n4 = np.expand_dims(util.get_nii_img(n4_path), axis=3)
    # # img_mask = np.expand_dims(util.get_nii_img(mask_path), axis=3) 

    # # Pad array since I stupidly used template with dimensions not factorable by 8
    # # Assumes input is (77, 91, 77) and pad to (80, 96, 80) with zeros
    # img_T1 = np.pad(img_T1, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    # # img_b0_d = np.pad(img_b0_d, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    # img_n4 = np.pad(img_n4, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')
    # # img_mask = np.pad(img_mask, ((2, 1), (3, 2), (2, 1), (0, 0)), 'constant')

    # # Convert to torch img format
    # img_T1 = util.nii2torch(img_T1)
    # # img_b0_d = util.nii2torch(img_b0_d)
    # img_n4 = util.nii2torch(img_n4)
    # # img_mask = util.nii2torch(img_mask) != 0

    # # Normalize data Should I normize my N4 output?
    # img_T1 = util.normalize_img(img_T1, 150, 0, 1, -1)  # Based on freesurfers T1 normalization
    # # max_img_b0_d = np.percentile(img_b0_d, 99)          # This usually makes majority of CSF be the upper bound
    # # min_img_b0_d = 0                                    # Assumes lower bound is zero (direct from scanner)
    # # img_b0_d = util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)
    # # img_b0_u = util.normalize_img(img_b0_u, max_img_b0_d, min_img_b0_d, 1, -1) # Use min() and max() from distorted data

    # # Set "data" and "target"
    # # img_data = np.concatenate((img_b0_d, img_T1), axis=1)
    # # img_target = img_b0_u

    # img_data =  img_T1
    # img_target = img_n4

    # # Send data to device
    # img_data = torch.from_numpy(img_data).float().to(device)
    # img_target = torch.from_numpy(img_target).float().to(device)
    # # img_mask = torch.from_numpy(np.array(img_mask, dtype=np.uint8))

    # return img_data, img_target #, img_mask