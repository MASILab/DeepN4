import nibabel as nib
import torch
import numpy as np
import math
import ants

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

def rmse(a,b):   
    MSE = np.square(np.subtract(a,b)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE


def init_parameters(self):
    '''
    initialize transformation parameters
    return random transformaion parameters
    '''
    self.init_config(self.config_dict)
    self._device = 'cuda' if self.use_gpu else 'cpu'

    self._dim = len(self.control_point_spacing)
    self.spacing = self.control_point_spacing
    self._dtype = torch.float32
    self.batch_size = self.data_size[0]
    self._image_size = np.array(self.data_size[2:])
    self.magnitude = self.epsilon
    assert 0<=self.magnitude<1, 'please set magnitude witihin [0,1)'
    self.order = self.interpolation_order
    self.downscale = self.downscale  # reduce image size to save memory

    self.use_log = True  if self.space == 'log' else False

    # contruct and initialize control points grid with random values
    self.param, self.interp_kernel = self.init_control_points_config()
    return self.param

def bspline_ants(k, data_cropped, axis, interp_mmap):
    number_of_random_points = 10000
    slice_idx = k
    if axis == 'z':
        img_array = data_cropped[:,:,slice_idx]
    if axis == 'y':
        img_array = data_cropped[:,slice_idx,:]
    if axis == 'x':
        img_array = data_cropped[slice_idx,:,:]

    row_indices = np.random.choice(range(2, img_array.shape[0]), number_of_random_points)
    col_indices = np.random.choice(range(2, img_array.shape[1]), number_of_random_points)
    scattered_data = np.zeros((number_of_random_points, 1))
    parametric_data = np.zeros((number_of_random_points, 2))
    for i in range(number_of_random_points):
        scattered_data[i,0] = img_array[row_indices[i], col_indices[i]]
        parametric_data[i,0] = row_indices[i]
        parametric_data[i,1] = col_indices[i]

    bspline_img = ants.fit_bspline_object_to_scattered_data(
        scattered_data, parametric_data,
        parametric_domain_origin=[0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0],
        parametric_domain_size = img_array.shape,
        number_of_fitting_levels=5, mesh_size=1)

    if axis == 'z':
        interp_mmap[:, :, k] = bspline_img.numpy()[:,:]
    if axis == 'y':
        interp_mmap[:, k, :] = bspline_img.numpy()[:,:]
    if axis == 'x':
        interp_mmap[k, :, :] = bspline_img.numpy()[:,:]