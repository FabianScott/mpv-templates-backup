import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import typing


def get_gausskernel_size(sigma, force_odd=True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2 == 0 and force_odd:
        ksize += 1
    return int(ksize)


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    '''Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2'''
    coeff = 1 / (math.sqrt(2 * math.pi) * sigma )
    exp = torch.exp(-((x ** 2) / (2 * sigma ** 2)))
    return coeff * exp


def gaussian_deriv1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    '''Function that computes values of a (1D) Gaussian derivative'''
    gauss = gaussian1d(x, sigma)
    exp = - x / (sigma ** 2)
    return gauss * exp


def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    kernel = kernel.flip((-2, -1)).unsqueeze(0).unsqueeze(0)

    # Apply convolution with padding to keep the output size the same
    x_pad = F.pad(x, (kernel.shape[-1] // 2,) * 4, mode='replicate')
    output = F.conv2d(x_pad, kernel)

    return output
    ## Do not forget about flipping the kernel!
    ## See in details here https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5


def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        
    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """
    ksize = get_gausskernel_size(sigma)
    x_pad = F.pad(x, (ksize // 2,) * 4, mode='replicate')

    # x_size: 1, 1, 1, ksize
    kernel_1d = gaussian1d(torch.arange(-ksize//2 + 1, ksize//2 + 1), sigma=sigma)
    # y_size: 1, 1, ksize, 1
    kernel = torch.outer(kernel_1d, kernel_1d).unsqueeze(0).unsqueeze(0)
    kernel = kernel / kernel.sum()
    out = F.conv2d(x_pad, kernel)
    return out


def spatial_gradient_first_order(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    b, c, h, w = x.shape
    ksize = get_gausskernel_size(sigma)
    x_gauss = gaussian_filter2d(x, sigma)

    kernel_grad_x = torch.tensor([1., 0, -1]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    kernel_grad_y = kernel_grad_x.view((-1, 1)).unsqueeze(0).unsqueeze(0)

    x_x = F.pad(x_gauss, (1, 1, 0, 0), mode='replicate')
    x_y = F.pad(x_gauss, (0, 0, 1, 1), mode='replicate')
    gx = F.conv2d(x_x, kernel_grad_x)
    gy = F.conv2d(x_y, kernel_grad_y)

    out = torch.empty((b, c, 2, h, w), dtype=x.dtype)
    out[:, :, 0] = gx
    out[:, :, 1] = gy
    return out


def affine(center: torch.Tensor, unitx: torch.Tensor, unity: torch.Tensor) -> torch.Tensor:
    r"""Computes transformation matrix A which transforms a point in homogeneous coordinates from the canonical coordinate system into image

    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 2)`, :math:`(B, 2)`, :math:`(B, 2)` 
        - Output: :math:`(B, 3, 3)`

    """
    assert center.size(0) == unitx.size(0)
    assert center.size(0) == unity.size(0)
    B = center.size(0)
    out = torch.zeros((B, 3, 3), dtype=unitx.dtype)
    out[:, :2, 0] = unitx - center
    out[:, :2, 1] = unity - center
    out[:, :2, 2] = center
    out[:, 2, 2] = 1

    return out


import torch
import torch.nn.functional as F


def extract_affine_patches(input: torch.Tensor,
                           A: torch.Tensor,
                           img_idxs: torch.Tensor,
                           PS: int = 32,
                           ext: float = 6.0):
    assert input.size(0) > 0
    b, ch, h, w = input.size()
    num_patches = A.size(0)

    # Generate coordinates for the output patches
    linspace = torch.linspace(-ext, ext, PS)
    grid_y, grid_x, = torch.meshgrid(linspace,  linspace)
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).repeat(num_patches, 1, 1, 1)  # (N, PS, PS, 3)

    transformed_grid = []
    for a, g in zip(A, grid):
        transformed_grid.append(g@a) # Apply transformation
    transformed_grid = torch.stack(transformed_grid)

    transformed_grid[:, :, :, 0] = transformed_grid[:, :, :, 0]/w
    transformed_grid[:, :, :, 1] = transformed_grid[:, :, :, 1]/h
    imgs = input[img_idxs].squeeze(1)   # Why is there an extra dimension when indexing?

    patches = F.grid_sample(imgs, transformed_grid[..., :2], align_corners=True)

    return patches


def extract_antializased_affine_patches(input: torch.Tensor,
                                        A: torch.Tensor,
                                        img_idxs: torch.Tensor,
                                        PS: int = 32,
                                        ext: float = 6.0):
    """Extract patches defined by affine transformations A from scale pyramid created image tensor X.
    It runs your implementation of the `extract_affine_patches` function, so it would not work w/o it.
    You do not need to ever modify this finction, implement `extract_affine_patches` instead.
    
    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    import kornia
    b, ch, h, w = input.size()
    num_patches = A.size(0)
    scale = (kornia.feature.get_laf_scale(ext * A.unsqueeze(0)[:, :, :2, :]) / float(PS))[0]
    half: float = 0.5
    pyr_idx = (scale.log2()).relu().long()
    cur_img = input
    cur_pyr_level = 0
    out = torch.zeros(num_patches, ch, PS, PS).to(device=A.device, dtype=A.dtype)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        _, ch_cur, h_cur, w_cur = cur_img.size()
        scale_mask = (pyr_idx == cur_pyr_level).squeeze()
        if (scale_mask.float().sum()) > 0:
            scale_mask = (scale_mask > 0).view(-1)
            current_A = A[scale_mask]
            current_A[:, :2, :3] *= (float(h_cur) / float(h))
            patches = extract_affine_patches(cur_img,
                                             current_A,
                                             img_idxs[scale_mask],
                                             PS, ext)
            out.masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.geometry.pyrdown(cur_img)
        cur_pyr_level += 1
    return out
