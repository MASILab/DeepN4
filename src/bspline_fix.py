import torch
import numpy as np
from torch.nn import functional as F

def bspline_kernel_3d_redo(
    spacing_between_knots=[2,2,2], order=3, asTensor=False, dtype=torch.float32, device=None
    ):

    kernel_ones = torch.ones(1, 1, *spacing_between_knots)
    kernel = kernel_ones
    #kernel = torch.ones(1, 1, 1, 1, 1)
    padding = np.array(spacing_between_knots) - 1
    
    for i in range(1, order):
        # change 2d to 3d
        kernel = F.conv3d(
            kernel,
            kernel_ones,
            padding=padding.tolist()
        ) / ( torch.prod(torch.tensor(spacing_between_knots)) )

    if asTensor and device is not None:
        return kernel.to(dtype=dtype, device=device)
    elif asTensor:
        return kernel.to(dtype=dtype)
    else:
        return kernel.numpy()



def bspline_kernel_2d_redo(
    spacing_between_knots=[2,2], order=3, asTensor=False, dtype=torch.float32, device=None
    ):
    kernel_ones = torch.ones(1, 1, *spacing_between_knots)
    kernel = kernel_ones
    #kernel = torch.ones(1, 1, 1, 1, 1)
    padding = np.array(spacing_between_knots)
    
    for i in range(1, order):
        # change 2d to 3d
        print(kernel.shape)
        kernel = F.conv2d(
            kernel,
            kernel_ones,
            padding=padding.tolist()
        ) / ( torch.prod(torch.tensor(spacing_between_knots)) )
        print(kernel.shape)

    if asTensor and device is not None:
        return kernel.to(dtype=dtype, device=device)
    elif asTensor:
        return kernel.to(dtype=dtype)
    else:
        return kernel.numpy()

