import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def hdist(H: torch.Tensor, pts_matches: torch.Tensor):
    '''Function, calculates one-way reprojection error
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error


    Shape:
        - Input :math:`(3, 3)`, :math:`(B, 4)`
        - Output: :math:`(B, 1)`
    '''
    B, dim = pts_matches.shape
    # The original coords
    vec = torch.concat((pts_matches[:, :2], torch.ones((B, 1))), dim=1)
    # The projected coords
    vec_ = pts_matches[:, 2:]
    H_vec = H @ vec

    dist = torch.cdist(vec_, H_vec)

    return dist


def sample(pts_matches: torch.Tensor, num: int = 4):
    '''Function, which draws random sample from pts_matches
    
    Return:
        torch.Tensor:

    Args:
        pts_matches: torch.Tensor: 2d tensor
        num (int): number of correspondences to sample

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(num, 4)`
    '''
    indices = torch.randperm(pts_matches.size(0))
    sample = pts_matches[indices[:num]]
    return sample


def getH(min_sample):
    '''Function, which estimates homography from minimal sample
    Return:
        torch.Tensor:

    Args:
        min_sample: torch.Tensor: 2d tensor

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(3, 3)`
    '''
    # Construct the matrix C
    C = torch.zeros(8, 9)
    for i in range(4):
        x, y, xp, yp = min_sample[i]
        C[2*i] = torch.tensor([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        C[2*i + 1] = torch.tensor([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    # Compute the null space of C
    _, _, V = torch.svd(C)
    H = V[-1].reshape(3, 3)
    H = H/H[:, -1,-1]
    # Check if some triplets lie close to a line
    if torch.matrix_rank(C) < 8:
        return None
    else:
        return H

    H_norm = torch.eye(3)
    return H_norm


def nsamples(n_inl: int, num_tc: int, sample_size: int, conf: float):
    return torch.log(1-conf) / torch.log(1-(n_inl ** sample_size))


def ransac_h(pts_matches: torch.Tensor, th: float = 4.0, conf: float = 0.99, max_iter: int = 1000):
    '''Function, which robustly estimates homography from noisy correspondences
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error

    Args:
        pts_matches: torch.Tensor: 2d tensor
        th (float): pixel threshold for correspondence to be counted as inlier
        conf (float): confidence
        max_iter (int): maximum iteration, overrides confidence
        
    Shape:
        - Input  :math:`(B, 4)`
        - Output: :math:`(3, 3)`,   :math:`(B, 1)`
    '''
    i = 0
    sample_size = 4
    H_best, support_best = torch.eye(3), -torch.inf

    while i < max_iter:
        current_points = sample(pts_matches, sample_size)
        H = getH(current_points)
        if H is not None:
            support = hdist(H, pts_matches)
            if torch.sum(support) > torch.sum(support_best):
                H_best, support_best = H, support
            max_iter = nsamples(n_inl=torch.sum(support_best > th).item(), num_tc=0, sample_size=sample_size, conf=conf)

    inl = support_best > th
    return H_best, inl
