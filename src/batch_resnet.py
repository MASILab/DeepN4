import torch
from torch.nn.functional import interpolate
import numpy as np
from tqdm import tqdm
from utils import unnormalize_img, rmse
import nibabel as nib
import interpol
from scipy.ndimage import spline_filter
from pathlib import Path
import os
import nibabel.processing as proc
from scipy.ndimage import gaussian_filter
from pytorch_msssim import SSIM
import scipy.ndimage as nd

# from loss import GeneratorLoss

# genloss = GeneratorLoss().cuda()
def log_gabor_3d(shape, sigma, frequency, theta, phi):
    """
    Create a 3D log-Gabor filter with the specified shape, standard deviation,
    frequency, orientation, and phase.
    """
    sigma_x = sigma[0]
    sigma_y = sigma[1]
    sigma_z = sigma[2]
    
    # Create 3D meshgrid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    
    # Convert x, y, z to spherical coordinates
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / rho)
    
    # Rotate coordinates by the specified angle
    theta = theta - theta
    phi = phi - phi
    
    # Convert back to Cartesian coordinates
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    
    # Compute the log-Gabor filter
    loggabor3d = np.exp(-0.5 * ((x / sigma_x)**2 + (y / sigma_y)**2 + (z / sigma_z)**2)) * np.cos(2 * np.pi * frequency * x + phi)
    return loggabor3d

def train(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    #with tqdm(total=len(loader)) as pbar:
    with tqdm(total=len(loader.dataset.train_iter)) as pbar:
        for batch_idx, sample in enumerate(loader):
            in_features, correct, bias = sample['input'], sample['correct'], sample['bias']
            in_features, correct, bias = in_features.to(device), correct.to(device), bias.to(device)
            in_squeeze = in_features.squeeze()
            in_numpy = (in_squeeze.cpu()).numpy()

            loggabor = log_gabor_3d((24,24,24), (4, 4, 4), 0, 0, 0)
            filtered_image1 = nd.convolve(in_numpy, loggabor)
            loggabor = log_gabor_3d((16,16,16), (4, 4, 4), 0, 0, 0)
            filtered_image2 = nd.convolve(in_numpy, loggabor)
            loggabor = log_gabor_3d((8,8,8), (4, 4, 4), 0, 0, 0)
            filtered_image3 = nd.convolve(in_numpy, loggabor)
            filtered_image1 = torch.Tensor(filtered_image1[np.newaxis,np.newaxis,...])
            filtered_image2 = torch.Tensor(filtered_image2[np.newaxis,np.newaxis,...])
            filtered_image3 = torch.Tensor(filtered_image3[np.newaxis,np.newaxis,...])

            optimizer.zero_grad()
            field = model(in_features,filtered_image1.to(device),filtered_image2.to(device),filtered_image3.to(device))
            pred = in_features / field
            # loss_fun = torch.nn.L1Loss()
            loss_fun = torch.nn.MSELoss() 
            alpha = 1
            beta = 0.01
            ssim_loss = SSIM(win_size=11, data_range=1,channel=108)
            loss = (beta * (1-ssim_loss(field[0,...], bias[0,...])) ) + (alpha * loss_fun(pred, bias) )


            mask = torch.Tensor(in_features > 0)
            mask = mask / torch.sum(mask)
            loss_masked = torch.sum(loss * mask)

            loss_masked.backward()
            optimizer.step()
            total_loss += loss_masked.item()

            pbar.set_description("Epoch {}\tAvg Loss: {:.4f}".format(epoch, total_loss / (batch_idx + 1)))
            pbar.update(1)
            torch.cuda.empty_cache()

        return total_loss / len(loader.dataset.train_iter)#len(loader)

def test(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for batch_idx, sample in enumerate(loader):
                in_features, correct, bias = sample['input'], sample['correct'], sample['bias']
                in_features, correct, bias = in_features.to(device), correct.to(device), bias.to(device)


                logfield = model(in_features,in_features,in_features,in_features)
                logpred = torch.log(in_features) - logfield
                loss_fun = torch.nn.MSELoss() 
                ssim_loss = SSIM(data_range=1, size_average=False)
                loss = ssim_loss(torch.exp(logfield), bias) + loss_fun(torch.exp(logpred), correct) 
                
                mask = torch.Tensor(in_features > 0)
                mask = mask / torch.sum(mask)
                loss_masked = torch.sum(loss * mask)

                #loss_fun = torch.nn.MSELoss(reduction='mean')
                # loss = loss_fun(output[mask==1], target[mask==1])
                # total_loss += loss.item()
                # total_loss += loss_masked.item()
                total_loss += loss_masked.item()

                
                # accuracy = dsc_fun(output, target)
                # total_accuracy += accuracy
                #TO DO total accracy / len

                pbar.set_description("  Test  \tAvg Loss: {:.4f}".format(total_loss / (batch_idx + 1)))
                pbar.update(1)

            # print(total_accuracy / len(loader.dataset.train_iter)  )  
            return total_loss / len(loader)#.dataset.train_iter)#len(loader)

def predict(model, loader, device, nii_path, out_path, out_bias_path, est_bias_path):
    model.eval()
    total_loss = 0
    outputs = []
    # starts = []
    # stops = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for batch_idx, sample in enumerate(loader):
                in_features, correct, bias = sample['input'], sample['correct'], sample['bias']
                in_features, correct, bias = in_features.to(device), correct.to(device), bias.to(device)

                estimated = bias
                logfield = model(in_features)
                logpred = torch.log(in_features) - logfield
                loss_fun = torch.nn.MSELoss(reduction='mean')
                loss =  loss_fun(torch.exp(logfield), bias)  + loss_fun(torch.exp(logpred), correct) 

                field = torch.exp(logfield)
                # field = interpol.spline_coeff_nd(field,interpolation=3,bound='dct2',dim=5,inplace=True)
                # field = spline_filter(field, order=3)
                # mask = torch.Tensor(in_features > 0)
                # mask = mask / torch.sum(mask)
                # loss_masked = loss * mask

                # mask[mask!=0.0] = 1.0
                # estimated = estimated * mask
                # field = mask * field
                total_loss += loss.item()

                pbar.set_description("  Test  \tAvg Loss: {:.4f}".format(total_loss / (batch_idx + 1)))
                pbar.update(1)

                in_features = in_features.cpu()  
                field = field.cpu()
                estimated = estimated.cpu()
                # outputs.append(logfield.clone())


            
            pad_idx, orig_shape = sample['pad_idx'], sample['orig_shape']
            lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ = pad_idx
            # out = torch.stack(outputs, dim=0).squeeze()
            # out = out.numpy() 
            field = field.squeeze()
            # field = field.numpy()
            estimated = estimated.squeeze()
            estimated = estimated.numpy()
            input_data = sample['input'].squeeze()
            # mask = mask.squeeze()
            # mask = mask.numpy()
            # loss_masked = loss_masked.squeeze()
            # loss_masked = loss_masked.numpy()

            # rmse_image = rmse(output, correct.cpu())
            # print('RMSE image', rmse_image)
            # rmse_field = rmse(estimated, field)
            # print('RMSE field', rmse_field) 
            
            # shape back to orginal image and unnormalize
            # final_out = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            # final_out[rx:rX,ry:rY,rz:rZ] = out[lx:lX,ly:lY,lz:lZ]
            final_field = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_field[rx:rX,ry:rY,rz:rZ] = field[lx:lX,ly:lY,lz:lZ]

            final_input = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_input[rx:rX,ry:rY,rz:rZ] = input_data[lx:lX,ly:lY,lz:lZ]

            # final_field = spline_filter(final_field, order=3)
            final_field = gaussian_filter(final_field, sigma=5)
            final_out = final_input / final_field

            # final_mask = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            # final_mask[rx:rX,ry:rY,rz:rZ] = mask[lx:lX,ly:lY,lz:lZ]
            # final_loss_masked = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            # final_loss_masked[rx:rX,ry:rY,rz:rZ] = loss_masked[lx:lX,ly:lY,lz:lZ]
            

            final_estimated = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_estimated[rx:rX,ry:rY,rz:rZ] = estimated[lx:lX,ly:lY,lz:lZ]
            # import pdb;pdb.set_trace()
            final_out = unnormalize_img(final_out, sample['max'].numpy(), 0, 1, 0)
            # final_mask = unnormalize_img(final_mask, sample['max'].numpy(), 0, 1, 0)
            # final_loss_masked = unnormalize_img(final_loss_masked, sample['max'].numpy(), 0, 1, 0)
            # final_field = unnormalize_img(final_field, sample['max'].numpy(), 0, 1, 0)
            # final_estimated = unnormalize_img(final_estimated, sample['max'].numpy(), 0, 1, 0)


            ref = nib.load(nii_path)
            nii = nib.Nifti1Image(final_out, affine=ref.affine, header=ref.header)
            nib.save(nii, out_path)

            nii = nib.Nifti1Image(final_field, affine=ref.affine, header=ref.header)
            nib.save(nii, out_bias_path)

            nii = nib.Nifti1Image(final_estimated, affine=ref.affine, header=ref.header)
            nib.save(nii, est_bias_path)

            # nii = nib.Nifti1Image(final_mask, affine=ref.affine, header=ref.header)
            # nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/src/unet_trained_model_bothloss/mask_3dunet_bothloss.nii.gz')

            # nii = nib.Nifti1Image(final_loss_masked, affine=ref.affine, header=ref.header)
            # nib.save(nii, '/nfs/masi/kanakap/projects/DeepN4/src/unet_trained_model_bothloss/lossmask_3dunet_bothloss.nii.gz')

            return total_loss / len(loader)#.dataset.train_iter)#len(loader)
