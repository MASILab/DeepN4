import torch
from torch.nn.functional import interpolate
import numpy as np
from tqdm import tqdm
from utils import save_nifti
import nibabel as nib
from pathlib import Path
import os
# from loss import GeneratorLoss

# genloss = GeneratorLoss().cuda()

def train(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    with tqdm(total=len(loader)) as pbar:
        for batch_idx, sample in enumerate(loader):
            in_features, target = sample['input'], sample['target']#, sample['mask']
            in_features, target = in_features.to(device), target.to(device)#, mask.to(device)

            optimizer.zero_grad()

            output = model(in_features)

            loss_fun = torch.nn.MSELoss(reduction='sum')
            # loss_fun = torch.nn.L1Loss()
            loss = loss_fun(output, target)
            #loss = genloss(output, target, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_description("Epoch {}\tAvg Loss: {:.4f}".format(epoch, total_loss / (batch_idx + 1)))
            pbar.update(1)

        return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for batch_idx, sample in enumerate(loader):
                in_features, target = sample['input'], sample['target']#, sample['mask']
                in_features, target = in_features.to(device), target.to(device)#, mask.to(device)

                output = model(in_features)

                # loss_fun = torch.nn.MSELoss(reduction='sum')
                loss_fun = torch.nn.L1Loss()
                # loss = loss_fun(output[mask==1], target[mask==1])
                loss = loss_fun(output, target)


                total_loss += loss.item()

                pbar.set_description("  Test  \tAvg Loss: {:.4f}".format(total_loss / (batch_idx + 1)))
                pbar.update(1)

            return total_loss / len(loader)

def predict(model, loader, device, nii_path, out_path):
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

                output = model(in_features)
                # outputs.append(output.cpu())

                # loss_fun = torch.nn.MSELoss(reduction='sum')
                loss_fun = torch.nn.L1Loss()
                loss = loss_fun(output, target)

                total_loss += loss.item()

                pbar.set_description("  Test  \tAvg Loss: {:.4f}".format(total_loss / (batch_idx + 1)))
                pbar.update(1)

                # xyz = sample['xyz']
                # overlap = sample['overlap']
                scale = sample['max']
                output = output.cpu()
                output = output * scale #* mask.cpu()
                # tmp = torch.zeros(overlap.shape)
                #
                # for i in range(len(xyz)):
                #     tmp[:,:,xyz[i][0]:xyz[i][0]+32, xyz[i][1]:xyz[i][1]+32, xyz[i][2]:xyz[i][2]+32] += output[i,:,:,:,:]
                # tmp = tmp / overlap
                # tmp = tmp * scale
                #
                outputs.append(output.clone())



            pad_idx, orig_shape = sample['pad_idx'], sample['orig_shape']
            # outputs[0] = outputs[0].squeeze()
            lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ = pad_idx
            # out = torch.cat(outputs, dim=0).squeeze()
            out = torch.stack(outputs, dim=0).squeeze()
            out = out.permute(1,2,3,0).numpy()
            final_out = np.zeros([orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]])
            final_out[rx:rX,ry:rY,rz:rZ,:] = out[lx:lX,ly:lY,lz:lZ,:]

            # outputs = torch.cat(outputs, dim=0)
            # out = outputs.squeeze().permute(1,2,3,0).numpy()

            ref = nib.load(nii_path)

            nii = nib.Nifti1Image(final_out, affine=ref.affine, header=ref.header)
            nib.save(nii, out_path)

            return total_loss / len(loader)
