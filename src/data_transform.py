import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
from numpy import random

import os

class dataset(Dataset):
    def __init__(self, csv, task=1):
        self.train_iter = []
        self.i = 0

        with open(csv, 'r') as f:
            for l in f.readlines():
                paths = l.strip().split(',')
                self.train_iter.append({'target':paths[0], 'input':paths[task]})

    def pad(self, img, sz):

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

    def normalize_img(self, img, max_img, min_img, a_max, a_min):

        img = (img - min_img)/(max_img - min_img)
        img = np.clip(img, a_max=a_max, a_min=a_min)

        return img

    def transform( self, img):

        x = random.randint(low=30, high=180)
        rotated = rotate(img, angle=x)
        fliped = np.fliplr(rotated).copy()

        return fliped

    def load(self, subj):
        target = nib.load(subj['target'])
        self.input = nib.load(subj['input']).get_fdata()
        self.input = self.transform(self.input)
        self.input, _ = self.pad(self.input, 128)
        self.in_max = np.percentile(self.input[np.nonzero(self.input)], 99.99)
        self.input = self.normalize_img(self.input, self.in_max, 0, 1, 0)
        

        self.affine = target.affine
        self.header = target.header
        self.target_unnorm = target.get_fdata()
        self.orig_shape = target.shape
        self.target_unnorm = self.transform(self.target_unnorm)
        self.target, self.pad_idx = self.pad(self.target_unnorm, 128)
        self.target_max = np.percentile(self.target[np.nonzero(self.target)], 99.99)
        self.target = self.normalize_img(self.target, self.in_max, 0, 1, 0)
        

    def prep_input(self):
        input_vols = self.input
        target_vols =  self.target
        input_vols = np.expand_dims(input_vols, axis=0)
        target_vols = np.expand_dims(target_vols, axis=0)

        return torch.from_numpy(input_vols).float(), torch.from_numpy(target_vols).float()#, torch.from_numpy(mask_vols).float()

    def __len__(self):

        return len(self.train_iter)

    def __getitem__(self, i):

        if torch.is_tensor(i): i = i.tolist()
        self.load(self.train_iter[i])
        input_vols, target_vols = self.prep_input()

        return {'input':input_vols, 'target':target_vols}#, 'mask':mask_vols}

class dataset_predict(dataset):
    def __init__(self, paths, task=1):
        self.load({'target': paths[0], 'input': paths[task]})#, 'bval': paths[3], 'bvec': paths[4]})


    def prep_input(self, i):

        input_vols = np.zeros((1, 128, 128, 128))
        target_vols = np.zeros((1, 128, 128, 128))

        input_vols[0,:,:,:] = self.input
        target_vols[0,:,:,:] = self.target

        return torch.from_numpy(input_vols).float(), torch.from_numpy(target_vols).float(), self.in_max, self.target_unnorm

    def __len__(self):
        # return int(np.ceil((self.input.shape[3])/1))
        return 1

    def __getitem__(self,i):
        input_vols, target_vols, in_max, target_unnorm = self.prep_input(i)

        return {'input': input_vols, 'target': target_vols,
                'orig_shape': self.orig_shape,
                'max':in_max, 'pad_idx':self.pad_idx, 'target_unnorm': target_unnorm}
