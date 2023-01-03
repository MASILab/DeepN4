import torch
import numpy as np
import nibabel as nib
import nibabel.processing as proc
import pickle as pkl
from torch.utils.data import Dataset
from pathlib import Path
# from dipy.io import read_bvals_bvecs
import os

class dataset(Dataset):
    def __init__(self, csv, task=1):
        self.train_iter = []
        self.i = 0

        with open(csv, 'r') as f:
            for l in f.readlines():
                paths = l.strip().split(',')
                self.train_iter.append({'target':paths[0], 'input':paths[task]})#, 'bval':paths[3], 'bvec':paths[4]})

    def pad(self, img, sz):
        # tmp = np.zeros((sz, sz, sz, img.shape[3]))
        tmp = np.zeros((sz, sz, sz))

        diff = int((sz-img.shape[0])/2)
        lx = max(diff,0)
        lX = min(img.shape[0]+diff,sz)

        diff = (img.shape[0]-sz) / 2
        rx = max(int(np.floor(diff)),0)
        rX = min(img.shape[0]-int(np.ceil(diff)),img.shape[0])

        diff = int((sz - img.shape[1]) / 2)
        ly = max(diff, 0)
        lY = min(img.shape[1] + diff, sz)

        diff = (img.shape[1] - sz) / 2
        ry = max(int(np.floor(diff)), 0)
        rY = min(img.shape[1] - int(np.ceil(diff)), img.shape[1])

        diff = int((sz - img.shape[2]) / 2)
        lz = max(diff, 0)
        lZ = min(img.shape[2] + diff, sz)

        diff = (img.shape[2] - sz) / 2
        rz = max(int(np.floor(diff)), 0)
        rZ = min(img.shape[2] - int(np.ceil(diff)), img.shape[2])

        tmp[lx:lX,ly:lY,lz:lZ] = img[rx:rX,ry:rY,rz:rZ]

        return tmp, [lx,lX,ly,lY,lz,lZ,rx,rX,ry,rY,rz,rZ]

    def normalize_img(self, img, max_img, min_img, max, min):
    # Scale between [1 0]
        img = (img - min_img)/(max_img - min_img)

        # Scale between [max min]
        img = img*(max - min) + min

        return img

    def load(self, subj):
        target = nib.load(subj['target'])
        self.affine = target.affine
        self.header = target.header
        self.target = target.get_fdata()
        self.orig_shape = target.shape
        voxel_size = [2,2,2]

        self.resampled_target = proc.resample_to_output(target, voxel_size)
        self.target, self.pad_idx = self.pad(self.resampled_target.get_fdata(), 128)
        self.target = self.normalize_img(self.target, 150, 0, 0, 1)

        input_img= nib.load(subj['input'])
        self.input = nib.load(subj['input']).get_fdata()
        self.resampled_input = proc.resample_to_output(input_img, voxel_size)
        # self.mask = nib.load(os.path.join('/'.join(subj['input'].split('/')[0:-1]), 'seg', '5iso_seg.nii.gz')).get_fdata()

        self.input, _ = self.pad(self.resampled_input.get_fdata(), 128)
        self.input = self.normalize_img(self.input, 150, 0, 0, 1)

        # self.mask, _ = self.pad(np.expand_dims(self.mask, axis=3), 96)
        # self.mask = np.squeeze(self.mask)

        # self.bval, self.bvec = read_bvals_bvecs(subj['bval'], subj['bvec'])

    def prep_input(self):
        idxs = []
        # input_vols = np.zeros((1,32,32,32))
        # target_vols = np.zeros((1,32,32,32))
        # mask_vols = np.zeros((1, 96, 96, 96))
        input_vols = self.input
        target_vols =  self.target
        input_vols = np.expand_dims(input_vols, axis=0)
        target_vols = np.expand_dims(target_vols, axis=0)

        # x = np.random.choice(np.arange(0, self.input.shape[0]-16, 16))
        # y = np.random.choice(np.arange(0, self.input.shape[1]-16, 16))
        # z = np.random.choice(np.arange(0, self.input.shape[2]-16, 16))
        #
        # if x + 32 > self.input.shape[0]:
        #     x = self.input.shape[0] - 32
        # if y + 32 > self.input.shape[1]:
        #     y = self.input.shape[1] - 32
        # if z + 32 > self.input.shape[2]:
        #     z = self.input.shape[2] - 32

        # for i in range(1):
        #     #idxs.append(np.random.randint(0, self.input.shape[3]))
        #     idxs.append(np.random.randint(0,10))

        # for i,idx in enumerate(idxs):
        #     # vol = self.input[x:x+32, y:y+32, z:z+32, idx].copy()
        #     vol = self.input[:, :, :, idx].copy()
        #     # tmp = vol[self.mask==2]
        #     max = np.percentile(vol, 99.99)
        #     # vol = self.input[:, :, :, idx].copy()l
        #     # max = np.percentile(self.input[:, :, :, idx], 99.99)
        #     vol = vol/max


        #     # mu = np.mean(tmp)
        #     # std = np.std(tmp)
        #     # vol = (vol-mu)/std


        #     input_vols[i,:,:,:] = vol

        #     # vol = self.target[x:x+32, y:y+32, z:z+32, idx].copy()
        #     vol = self.target[:, :, :, idx].copy()
        #     # max = np.percentile(self.target[:, :, :, idx], 99.99)
        #     vol = vol/max
        #     # vol = (vol - mu) / std
        #     target_vols[i,:,:,:] = vol

            # mask_vols[i,:,:,:] = self.mask[:,:,:].copy()
            # mask_vols[mask_vols>0] = 1

        return torch.from_numpy(input_vols).float(), torch.from_numpy(target_vols).float()#, torch.from_numpy(mask_vols).float()

    def __len__(self):
        # return int(len(self.train_iter)/10)*3
        return len(self.train_iter)
        #return 1344

    def __getitem__(self, i):
        if torch.is_tensor(i): i = i.tolist()
        self.load(self.train_iter[i])

        # if self.i%self.__len__() == 0:
        #     idx = np.random.randint(0, len(self.train_iter))
        #     self.load(self.train_iter[idx])
        # self.i += 1

        # # input_vols, target_vols, mask_vols = self.prep_input()
        input_vols, target_vols = self.prep_input()

        return {'input':input_vols, 'target':target_vols}#, 'mask':mask_vols}

class dataset_predict(dataset):
    def __init__(self, paths, task=1):
        self.load({'target': paths[0], 'input': paths[task]})#, 'bval': paths[3], 'bvec': paths[4]})

    def prep_input(self, i):
        idxs = []


        # input_vols = np.zeros((1, 48, 48, 48))
        # target_vols = np.zeros((1, 96, 96, 96))

        x,y,z = 0, 0, 0

        idxs = [i]
        start = i
        stop = i

        # if (i+1)*30 < self.input.shape[3]:
        #     idxs.extend(np.arange(i*30+1, (i+1)*30))
        #     start = i*30+1
        #     stop = (i+1)*30
        # else:
        #     idxs.extend(np.arange(self.input.shape[3]-31, self.input.shape[3]-1))
        #     start = self.input.shape[3]-31
        #     stop = self.input.shape[3]-1

        # mask_vols = np.zeros((1, 96, 96, 96))
        input_vols = np.zeros((1, 96, 96, 96))
        target_vols = np.zeros((1, 96, 96, 96))

        # xyz = []
        # overlap = np.zeros((1,self.input.shape[0], self.input.shape[1], self.input.shape[2]))

        # tmp = self.input[:, :, :, i]
        # tmp = tmp[self.mask == 2]
        # inmax = np.percentile(tmp, 99.99)

        inmax = np.percentile(self.input[:, :, :], 99.99)
        # targmax = np.percentile(self.target[:, :, :, i], 99.99)


        # vol = self.input[x:x+32, y:y+32, z:z+32, idx].copy()
        # vol = self.input[:, :, :, i].copy()
        # vol = self.input[:, :, :, idx].copy()l
        # max = np.percentile(self.input[:, :, :, idx], 99.99)
        # vol = vol/inmax

        input_vols[:,:,:] = self.input

        # vol = self.target[:, :, :, i].copy()
        # vol = vol/inmax
        target_vols[:,:,:] = self.target

        # mask_vols[0,:,:,:] = self.mask[:,:,:].copy()
        # mask_vols[mask_vols>0] = 1

        # return torch.from_numpy(input_vols).float(), torch.from_numpy(target_vols).float(), torch.from_numpy(mask_vols).float(), inmax
        return torch.from_numpy(input_vols).float(), torch.from_numpy(target_vols).float(), inmax

        # for x in range(0,self.input.shape[0]-16,16):
        #     for y in range(0, self.input.shape[1]-16, 16):
        #         for z in range(0, self.input.shape[2]-16, 16):
        #             input_vol = np.zeros((1,32,32,32))
        #             target_vol = np.zeros((1,32,32,32))
        #             for i,idx in enumerate(idxs):
        #
        #                 ix, iy, iz = x, y, z
        #                 if ix+32 > self.input.shape[0]:
        #                     ix = self.input.shape[0]-32
        #                 if iy+32 > self.input.shape[1]:
        #                     iy = self.input.shape[1]-32
        #                 if iz + 32 > self.input.shape[2]:
        #                     iz = self.input.shape[2] - 32
        #
        #                 # vol = self.input[:, :, :, idx].copy()
        #                 vol = self.input[ix:ix + 32, iy:iy + 32, iz:iz + 32, idx].copy()
        #
        #                 vol = vol/inmax
        #                 input_vol[i,:,:,:] = vol
        #
        #                 # vol = self.target[:, :, :, idx].copy()
        #                 vol = self.target[ix:ix + 32, iy:iy + 32, iz:iz + 32, idx].copy()
        #
        #                 vol = vol/targmax
        #                 target_vol[i,:,:,:] = vol
        #
        #                 overlap[i, ix:ix + 32, iy:iy + 32, iz:iz + 32] += 1
        #                 xyz.append([ix, iy, iz])
        #
        #                 input_vols.append(torch.from_numpy(input_vol).float())
        #                 target_vols.append(torch.from_numpy(target_vol).float())

        # return input_vols, target_vols, overlap, inmax


    def __len__(self):
        # return int(np.ceil((self.input.shape[3])/1))
        return 1

    def __getitem__(self,i):
        input_vols, target_vols, inmax = self.prep_input(i)

        return {'input': input_vols, 'target': target_vols,
                'orig_shape': self.orig_shape,
                'max':inmax, 'pad_idx':self.pad_idx}
