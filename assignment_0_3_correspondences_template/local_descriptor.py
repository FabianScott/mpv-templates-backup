import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import *
from local_detector import *
from tqdm import tqdm

def affine_from_location(b_ch_d_y_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)` 
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """

    A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    A[:, 0, 0] = b_ch_d_y_x[:, 2]
    A[:, 1, 1] = b_ch_d_y_x[:, 2]
    A[:, 0, 2] = b_ch_d_y_x[:, 4]
    A[:, 1, 2] = b_ch_d_y_x[:, 3]
    A[:, 2, 2] = 1

    img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()
    img_idxs[:, 0] = b_ch_d_y_x[:, 0]
    return A, img_idxs


def affine_from_location_and_orientation(b_ch_d_y_x: torch.Tensor,
                                         ori: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian). Ori - orientation angle in radians
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1) 
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    A, img_idxs = affine_from_location(b_ch_d_y_x)
    rotational_mat = torch.zeros_like(A)
    rotational_mat[:, 0, 0] = torch.cos(ori).flatten()
    rotational_mat[:, 1, 1] = torch.cos(ori).flatten()
    rotational_mat[:, 0, 1] = torch.sin(ori).flatten()
    rotational_mat[:, 1, 0] = -torch.sin(ori).flatten()
    rotational_mat[:, 2, 2] = 1
    # A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    # img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()
    return A @ rotational_mat, img_idxs


def sqrtm(mat):
    U, S_vals, V = torch.svd(mat)
    S_sqrt = torch.zeros_like(mat)
    S_sqrt[:, torch.arange(2), torch.arange(2)] = S_vals ** .5
    return S_sqrt


def affine_from_location_and_orientation_and_affshape(b_ch_d_y_x: torch.Tensor,
                                                      ori: torch.Tensor,
                                                      aff_shape: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1), :math:`(B, 3)
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    A, img_idxs = affine_from_location_and_orientation(b_ch_d_y_x, ori)

    C = torch.zeros((A.size(0), 2, 2))
    for i, row in enumerate(aff_shape):
        a, b, c = row
        C[i, :, :] = torch.tensor([[a, c], [c, b]])

    sqrt_C = sqrtm(C)
    inv_sqrt_C = torch.linalg.inv(sqrt_C)
    det_sqrt_inv_C = torch.det(inv_sqrt_C)
    sqrt_det = torch.sqrt(det_sqrt_inv_C)
    A[:, :2, :2] = torch.linalg.inv(inv_sqrt_C/sqrt_det)
    return A, img_idxs


def estimate_patch_dominant_orientation(x: torch.Tensor, num_angular_bins: int = 36):
    """Function, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.
    
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
        num_angular_bins: int, default is 36
    
    Returns:
        angles: (torch.Tensor) in radians shape [Bx1]
    """
    temp = spatial_gradient_first_order(x=x, sigma=1)
    Ix, Iy = temp[:, :, 0], temp[:, :, 1]
    angle = torch.atan2(Iy, Ix + 1e-10) + (torch.pi/2)
    out = torch.histogram(angle % torch.pi, bins=num_angular_bins)
    index = out.hist.argmax()
    return out.bin_edges[index]


def estimate_patch_affine_shape(x: torch.Tensor):
    """Function, which estimates the patch affine shape by second moment matrix. Returns ellipse parameters: a, b, c
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
    
    Returns:
        ell: (torch.Tensor) in radians shape [Bx3]
    """
    mean_x = torch.mean(x, dim=(2, 3), keepdim=True)
    centered_x = x - mean_x
    second_moment = torch.matmul(centered_x, centered_x.transpose(2, 3)) / (x.size(2) * x.size(3))

    eigenvalues = torch.abs(torch.real(torch.linalg.eigvals(second_moment)))

    a = torch.sqrt(eigenvalues[..., 0])
    b = torch.sqrt(eigenvalues[..., 1])
    c = torch.sqrt(eigenvalues[..., 2])

    ell = torch.stack((a, b, c), dim=1)

    return torch.real(ell)


def calc_sift_descriptor(input: torch.Tensor,
                         num_ang_bins: int = 8,
                         num_spatial_bins: int = 4,
                         clipval: float = 0.2) -> torch.Tensor:
    '''    
    Args:
        x: torch.Tensor (B, 1, PS, PS)
        num_ang_bins: (int) Number of angular bins. (8 is default)
        num_spatial_bins: (int) Number of spatial bins (4 is default)
        clipval: (float) default 0.2
        
    Returns:
        Tensor: SIFT descriptor of the patches

    Shape:
        - Input: (B, 1, PS, PS)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2)
    '''
    B, _, patch_size, _ = input.shape
    temp = spatial_gradient_first_order(input, sigma=1.)
    Ix, Iy = temp[:, :, 0], temp[:, :, 1]
    magnitude = torch.sqrt(Ix ** 2 + Iy ** 2 + 1e-10).squeeze(1)
    orientation = torch.atan2(Iy, Ix + 1e-10).squeeze(1) % 2.0 * torch.pi

    # Compute spatial bins
    bin_size = math.ceil(patch_size / num_spatial_bins)

    # Initialize descriptor
    descriptor = torch.zeros((B, num_ang_bins, num_spatial_bins, num_spatial_bins))
    descriptor_ = torch.zeros((B, num_ang_bins, num_spatial_bins, num_spatial_bins))
    gauss_ = gaussian1d(torch.linspace(-bin_size/2, bin_size/2, num_ang_bins), sigma=1e2)
    weight_mat = torch.outer(gauss_, gauss_).squeeze(0).repeat(B, 1, 1)

    # Iterate over spatial bins
    for i in tqdm(range(num_spatial_bins), desc="Sifting"):
        for j in range(num_spatial_bins):
            # Define spatial bin boundaries
            x_min = int(i * bin_size)
            x_max = int((i + 1) * bin_size)
            y_min = int(j * bin_size)
            y_max = int((j + 1) * bin_size)

            bin_indicies = torch.floor(orientation[:, x_min:x_max, y_min:y_max] / (2 * math.pi / num_ang_bins)).to(torch.int)
            sub_patch = magnitude[:, x_min:x_max, y_min:y_max]

            for bin_ in range(num_ang_bins):
                mask = bin_indicies == bin_
                descriptor_[:, bin_, i, j] = torch.sum(sub_patch[mask] * weight_mat[mask])

            # Iterate over pixels in spatial bin
            # for x in range(x_min, x_max):
            #     for y in range(y_min, y_max):
            #         for b in range(B):
            #             # Compute gradient orientation bin
            #             bin_index = torch.floor(orientation[b, x, y] / (2 * math.pi / num_ang_bins)).to(torch.int)
            #             # Accumulate magnitude into descriptor
            #             # print(bin_index, i, j, magnitude[..., x, y], weight_mat[x-x_min, y-y_min], magnitude[..., x, y] * weight_mat[x-x_min, y-y_min])
            #             descriptor_[b, bin_index, i, j] += magnitude[b, x, y] * weight_mat[b, x-x_min, y-y_min]
            # print(i, j, torch.sum(torch.abs(descriptor - descriptor_))/torch.sum(descriptor))
    # L2-normalize descriptor
    descriptor = F.normalize(descriptor.view(B, -1), p=2, dim=1)
    # Clip descriptor values
    descriptor = torch.clamp(descriptor, max=clipval)
    # L2-normalize again
    descriptor = F.normalize(descriptor, p=2, dim=1)

    return descriptor


def photonorm(x: torch.Tensor):
    """Function, which normalizes the patches such that the mean intensity value per channel will be 0 and the standard deviation will be 1.0. Values outside the range < -3,3> will be set to -3 or 3 respectively
    Args:
        x: (torch.Tensor) shape [BxCHxHxW]
    
    Returns:
        out: (torch.Tensor) shape [BxCHxHxW]
    """
    b, ch, h, w = x.size()
    out = x
    for c in range(ch):
        out[:, c] /= out[:, c].max() if out[:, c].max() else 1
    out = torch.clamp(out, min=-3, max=3)
    return out


if __name__ == '__main__':
    patch = torch.zeros(1, 1, 32, 32)
    patch[:, :, 16:, :] = 1.0

    num_ang_bins = 8
    num_spatial_bins = 4

    desc = calc_sift_descriptor(patch, num_ang_bins, num_spatial_bins)
