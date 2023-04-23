import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
from numpy import random
from custom_image3d10 import *

class dataset(Dataset):
    def __init__(self, csv, task=1):
        self.train_iter = []
        self.i = 0

        with open(csv, 'r') as f:
            for l in f.readlines():
                paths = l.strip().split(',')
                self.train_iter.append({'correct':paths[0], 'input':paths[task], 'bias':paths[2], 'filtered_image1':paths[3], 'filtered_image2':paths[4], 'filtered_image3':paths[5]})

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
        correct = nib.load(subj['correct'])
        bias = nib.load(subj['bias'])
        self.input = nib.load(subj['input']).get_fdata()

        
        # self.input = self.transform(self.input)
        self.input, _ = self.pad(self.input, 108)
        self.in_max = np.percentile(self.input[np.nonzero(self.input)], 99.99)
        self.input = self.normalize_img(self.input, self.in_max, 0, 1, 0)

        self.affine = correct.affine
        self.header = correct.header
        self.correct = correct.get_fdata()
        self.orig_shape = correct.shape
        # self.target_unnorm = self.transform(self.target_unnorm)
        self.correct, self.pad_idx = self.pad(self.correct, 108)
        self.correct_max = np.percentile(self.correct[np.nonzero(self.correct)], 99.99)
        self.correct = self.normalize_img(self.correct, self.in_max, 0, 1, 0)

        self.affine = bias.affine
        self.header = bias.header
        self.bias = bias.get_fdata()
        self.orig_shape = bias.shape
        # self.target_unnorm = self.transform(self.target_unnorm)
        self.bias, self.pad_idx = self.pad(self.bias, 108)

        self.f1 = nib.load(subj['filtered_image1']).get_fdata()
        self.f1, _= self.pad(self.f1, 108)
        self.f1 = self.normalize_img(self.f1, self.in_max, 0, 1, 0)

        self.f2 = nib.load(subj['filtered_image2']).get_fdata()
        self.f2, _= self.pad(self.f2, 108)
        self.f2 = self.normalize_img(self.f2, self.in_max, 0, 1, 0)

        self.f3 = nib.load(subj['filtered_image3']).get_fdata()
        self.f3, _ = self.pad(self.f3, 108)
        self.f3 = self.normalize_img(self.f3, self.in_max, 0, 1, 0)

    def prep_input(self):
        input_vols = self.input
        correct_vols =  self.correct
        bias_vols =  self.bias
        f1_vols =  self.f1
        f2_vols =  self.f2
        f3_vols =  self.f3
        input_vols = np.expand_dims(input_vols, axis=0)
        correct_vols = np.expand_dims(correct_vols, axis=0)
        bias_vols = np.expand_dims(bias_vols, axis=0)
        f1_vols = np.expand_dims(f1_vols, axis=0)
        f2_vols = np.expand_dims(f2_vols, axis=0)
        f3_vols = np.expand_dims(f3_vols, axis=0)

        return torch.from_numpy(input_vols).float(), torch.from_numpy(correct_vols).float(), torch.from_numpy(bias_vols).float(), torch.from_numpy(f1_vols).float(), torch.from_numpy(f2_vols).float(), torch.from_numpy(f3_vols).float()

    def __len__(self):

        return len(self.train_iter)

    def __getitem__(self, i):

        if torch.is_tensor(i): i = i.tolist()
        self.load(self.train_iter[i])
        input_vols, correct_vols, bias_vols, f1_vols, f2_vols, f3_vols  = self.prep_input()

        return {'input':input_vols, 'correct':correct_vols, 'bias':bias_vols, 'f1':f1_vols, 'f2':f2_vols, 'f3':f3_vols}

class dataset_predict(dataset):
    def __init__(self, paths, task=1):
        self.load({'correct':paths[0], 'input':paths[task], 'bias':paths[2]})#, 'bval': paths[3], 'bvec': paths[4]})


    def prep_input(self, i):

        # input_vols = np.zeros((1, 128, 128, 128))
        # correct_vols = np.zeros((1, 128, 128, 128))
        # bias_vols = np.zeros((1, 128, 128, 128))
        input_vols = np.zeros((1, 108, 108, 108))
        correct_vols = np.zeros((1, 108, 108, 108))
        bias_vols = np.zeros((1, 108, 108, 108))


        input_vols[0,:,:,:] = self.input
        correct_vols[0,:,:,:] = self.correct
        bias_vols[0,:,:,:] = self.bias

        return torch.from_numpy(input_vols).float(), torch.from_numpy(correct_vols).float(), torch.from_numpy(bias_vols).float(), self.in_max

    def __len__(self):
        # return int(np.ceil((self.input.shape[3])/1))
        return 1

    def __getitem__(self,i):
        input_vols, correct_vols, bias_vols, in_max = self.prep_input(i)

        return {'input': input_vols, 'correct': correct_vols, 'bias': bias_vols,
                'orig_shape': self.orig_shape,
                'max':in_max, 'pad_idx':self.pad_idx}
