import torch
import numpy as np
from utils import *
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import gaussian_filter
import tempfile
import os
from joblib import Parallel, delayed

# from loss import GeneratorLoss

# genloss = GeneratorLoss().cuda()

def train(model, loader, optimizer, device, epoch, model_dir):
    model.train()
    total_loss = 0

    #with tqdm(total=len(loader)) as pbar:
    with tqdm(total=len(loader.dataset.train_iter)) as pbar:
        for batch_idx, sample in enumerate(loader):
            in_features, correct, bias = sample['input'], sample['correct'], sample['bias']
            in_features, correct, bias = in_features.to(device), correct.to(device), bias.to(device)

            optimizer.zero_grad()

            logfield = model(in_features,device)
            logpred = torch.log(in_features) - logfield
            loss_fun = torch.nn.MSELoss()
            loss = loss_fun(torch.exp(logfield), bias) + loss_fun(torch.exp(logpred), correct) 

            mask = torch.Tensor(in_features > 0)
            mask = mask / torch.sum(mask)
            loss_masked = torch.sum(loss * mask)

            loss_masked.backward()
            optimizer.step()
            total_loss += loss_masked.item()

            pbar.set_description("Epoch {}\tAvg Loss: {:.4f}".format(epoch, total_loss / (batch_idx + 1)))
            pbar.update(1)
            torch.cuda.empty_cache()

            if batch_idx % 3000 == 0 and len(loader) != 0:
                save_model(model, optimizer, '{}/checkpoint_epoch_{}_{}'.format(model_dir, epoch, batch_idx))

        return total_loss / len(loader)#len(loader.dataset.train_iter)

def test(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for batch_idx, sample in enumerate(loader):
                in_features, correct, bias = sample['input'], sample['correct'], sample['bias']
                in_features, correct, bias = in_features.to(device), correct.to(device), bias.to(device)


                logfield = model(in_features,device)
                logpred = torch.log(in_features) - logfield
                loss_fun = torch.nn.MSELoss()
                loss = loss_fun(torch.exp(logfield), bias) + loss_fun(torch.exp(logpred), correct) 
                
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
                logfield = model(in_features,device)
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
            path = tempfile.mkdtemp()
            interp_mmap_path = os.path.join(path,'interp_mmap.mmap')
            interp_mmap = np.memmap(interp_mmap_path, dtype=float, shape=(88,88,88), mode='w+')

            zaxis = range(final_field.shape[2])
            yaxis = range(final_field.shape[1])
            xaxis = range(final_field.shape[0])
            results = Parallel(n_jobs=15)(delayed(bspline_ants)(k,final_field, 'z', interp_mmap) for k in zaxis)
            results = Parallel(n_jobs=15)(delayed(bspline_ants)(k,final_field, 'y', interp_mmap) for k in yaxis)
            results = Parallel(n_jobs=15)(delayed(bspline_ants)(k,final_field, 'x', interp_mmap) for k in xaxis)
            final_field = interp_mmap
            # final_field = gaussian_filter(final_field, sigma=5)
            final_out = final_input / final_field
            

            final_estimated = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2]])
            final_estimated[rx:rX,ry:rY,rz:rZ] = estimated[lx:lX,ly:lY,lz:lZ]
            # import pdb;pdb.set_trace()
            final_out = unnormalize_img(final_out, sample['max'].numpy(), 0, 1, 0)


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
