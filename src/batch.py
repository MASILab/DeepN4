import torch
from torch.nn.functional import interpolate
import numpy as np
from tqdm import tqdm
from utils import unnormalize_img, rmse
import nibabel as nib
from pathlib import Path
import os
import nibabel.processing as proc
from scipy.ndimage import gaussian_filter
# from loss import GeneratorLoss

# genloss = GeneratorLoss().cuda()

def train(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    #with tqdm(total=len(loader)) as pbar:
    with tqdm(total=len(loader.dataset.train_iter)) as pbar:
        for batch_idx, sample in enumerate(loader):
            in_features, target = sample['input'], sample['target']#, sample['mask']
            in_features, target = in_features.to(device), target.to(device)#, mask.to(device)

            optimizer.zero_grad()

            output = model(in_features)

            # loss_fun = torch.nn.MSELoss(reduction='mean')
            loss_fun = torch.nn.L1Loss()
            # target = np.log(target) * in_features
            loss = loss_fun(output, target)
            #loss = genloss(output, target, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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
                in_features, target = sample['input'], sample['target']#, sample['mask']
                in_features, target = in_features.to(device), target.to(device)#, mask.to(device)
                
                output = model(in_features)

                #loss_fun = torch.nn.MSELoss(reduction='mean')
                loss_fun = torch.nn.L1Loss()
                # loss = loss_fun(output[mask==1], target[mask==1])
                loss = loss_fun(output, target)
                total_loss += loss.item()
                
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
                in_features, target = sample['input'], sample['target']#, sample['mask']
                # in_features = torch.cat(in_features, dim=0)
                # target = torch.cat(target, dim=0)
                in_features, target = in_features.to(device), target.to(device)#, mask.to(device)
                # starts.extend(sample['start'])
                # stops.extend(sample['stop'])
                estimated = in_features / target
                output = model(in_features)
                field = in_features / output
                # outputs.append(output.cpu())

                loss_fun = torch.nn.MSELoss(reduction='mean')
                #loss_fun = torch.nn.L1Loss()
                loss = loss_fun(output, target)

                total_loss += loss.item()

                pbar.set_description("  Test  \tAvg Loss: {:.4f}".format(total_loss / (batch_idx + 1)))
                pbar.update(1)

                # xyz = sample['xyz']
                # overlap = sample['overlap']
                output = output.cpu()
                output = output #* scale #* mask.cpu()
                
                output = output.cpu()  
                field = field.cpu()
                estimated = estimated.cpu()

                # tmp = torch.zeros(overlap.shape)
                #
                # for i in range(len(xyz)):
                #     tmp[:,:,xyz[i][0]:xyz[i][0]+32, xyz[i][1]:xyz[i][1]+32, xyz[i][2]:xyz[i][2]+32] += output[i,:,:,:,:]
                # tmp = tmp / overlap
                # tmp = tmp * scale
                #
                outputs.append(output.clone())


            # out = torch.stack(outputs, dim=0).squeeze()
            out = output.squeeze() 
            out = out.numpy() 

            field = field.squeeze()
            field = field.numpy()
            estimated = estimated.squeeze()
            estimated = estimated.numpy()
            
            rmse_image = rmse(output, target.cpu())
            print('RMSE image', rmse_image)
            # rmse_field = rmse(estimated, field)
            # print('RMSE field', rmse_field) 

            pad_idx, orig_shape = sample['pad_idx'], sample['orig_shape']
            lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ = pad_idx
            final_out = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_out[rx:rX,ry:rY,rz:rZ] = out[lx:lX,ly:lY,lz:lZ]
            final_out = unnormalize_img(final_out, sample['max'].numpy(), 0, 1, 0)

            final_field = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_field[rx:rX,ry:rY,rz:rZ] = field[lx:lX,ly:lY,lz:lZ]
            final_estimated = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_estimated[rx:rX,ry:rY,rz:rZ] = estimated[lx:lX,ly:lY,lz:lZ]

#            final_field = gaussian_filter(final_field, sigma=5)

            ref = nib.load(nii_path)

            nii = nib.Nifti1Image(final_out, affine=ref.affine, header=ref.header)
            nib.save(nii, out_path)

            nii = nib.Nifti1Image(final_field, affine=ref.affine, header=ref.header)
            nib.save(nii, out_bias_path)

            nii = nib.Nifti1Image(final_estimated, affine=ref.affine, header=ref.header)
            nib.save(nii, est_bias_path)

            return total_loss / len(loader)#.dataset.train_iter)#len(loader)
