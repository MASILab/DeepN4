import nibabel as nib
import torch
import numpy as np
import math

def save_nifti(x, save_path, nifti_path):
    nib_img = nib.Nifti1Image(x, nib.load(nifti_path).affine, nib.load(nifti_path).header)
    nib.save(nib_img, save_path)

def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
    
def unnormalize_img(img, max_img, min_img, max, min):
# Undoes normalize_img()
    img = (img - min)/(max - min)*(max_img - min_img) + min_img

    return img


def rmse(a,b):   
    MSE = np.square(np.subtract(a,b)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE