import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import * 
from local_detector import *


def affine_from_location(b_ch_d_y_x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
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
                                         ori: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
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
    rotational_mat[:, 0, 0] = torch.cos(ori)
    rotational_mat[:, 1, 1] = torch.cos(ori)
    rotational_mat[:, 0, 1] = torch.sin(ori)
    rotational_mat[:, 1, 0] = -torch.sin(ori)
    rotational_mat[:, 2, 2] = 1
    # A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    # img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()
    return A @ rotational_mat, img_idxs


def affine_from_location_and_orientation_and_affshape(b_ch_d_y_x: torch.Tensor,
                                                      ori: torch.Tensor,
                                                      aff_shape: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1), :math:`(B, 3)
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    A, img_idxs = affine_from_location_and_orientation(b_ch_d_y_x, ori)
    C = torch.zeros(A.size(0), 2, 2)
    C[:, :2, 0] = aff_shape[:2]
    C[:, :2, 1] = aff_shape[1:]
    inv_sqrt_C = torch.linalg.inv(torch.linalg.sqrtm(C))
    # Compute determinant of inverse square root of C
    det_sqrt_inv_C = torch.det(inv_sqrt_C)
    A[:, :2, :2] = torch.linalg.inv(inv_sqrt_C * torch.sqrt(det_sqrt_inv_C))
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
    out = spatial_gradient_first_order(x=x, sigma=1)
    Ix, Iy = out[:, :, 0], out[:, :, 1]
    angle = torch.atan2(Iy, Ix)
    out = torch.histogram(angle, bins=num_angular_bins)
    index = out.hist.argmax()
    return out.bin_edges[index]

def estimate_patch_affine_shape(x: torch.Tensor):
    """Function, which estimates the patch affine shape by second moment matrix. Returns ellipse parameters: a, b, c
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
    
    Returns:
        ell: (torch.Tensor) in radians shape [Bx3]
    """
    out = torch.zeros(x.size(0), 3)
    return out



    

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
    out = torch.zeros(input.size(0), num_ang_bins * num_spatial_bins ** 2)
    return out


def photonorm(x: torch.Tensor):
    """Function, which normalizes the patches such that the mean intensity value per channel will be 0 and the standard deviation will be 1.0. Values outside the range < -3,3> will be set to -3 or 3 respectively
    Args:
        x: (torch.Tensor) shape [BxCHxHxW]
    
    Returns:
        out: (torch.Tensor) shape [BxCHxHxW]
    """
    out = x
    return out



